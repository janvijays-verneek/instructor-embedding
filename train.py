# This script is based on the modification from https://github.com/huggingface/transformers
import logging
import os
import torch
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import hashlib
import functools

import datasets
import nltk  # Here to have a nice missing dependency error message early on

import transformers
from filelock import FileLock
from InstructorEmbedding import INSTRUCTOR
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from torch.utils.data import Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.utils.versions import require_version
from datasets import Dataset,DatasetDict
import torch
from torch import nn

import torch.distributed as dist
from transformers import utils
from transformers import trainer_utils
from typing import Dict, Union, Any

if utils.is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

if utils.is_apex_available():
    from apex import amp

    

check_min_version("4.20.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

def tensor_info(tensor):
    """Compute hash and statistics for a PyTorch tensor."""
    # Compute hash
    tensor_bytes = tensor.detach().cpu().numpy().tobytes()
    hash_val = hashlib.sha256(tensor_bytes).hexdigest()
    
    # Compute statistics
    mean_val = torch.mean(tensor).item()
    max_val = torch.max(tensor).item()
    min_val = torch.min(tensor).item()
    std_val = torch.std(tensor).item()
    
    return {
        "hash": hash_val,
        "mean": mean_val,
        "max": max_val,
        "min": min_val,
        "std": std_val
    }


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False

class InstructorTrainer(Seq2SeqTrainer):
    def _get_train_sampler(self) :
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        if self.args.world_size <= 1:
            return SequentialSampler(self.train_dataset)
        else:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed,
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        for task_id in inputs['task_name']:
            assert task_id==inputs['task_name'][0],f"Examples in the same batch should come from the same task, " \
                                                    f"but task {task_id} and task {inputs['task_name'][0]} are found"
        cur_results = {}
        for k in ['query', 'pos', 'neg']:
            cur_inputs = {
                'input_ids': inputs[f'{k}_input_ids'],
                'attention_mask': inputs[f'{k}_attention_mask'],
                'context_masks': inputs[f'{k}_context_masks'],
            }
            cur_results[k] = model(cur_inputs)['sentence_embedding']

        embeddings_query, embeddings_pos, embeddings_neg, cl_temperature = \
            cur_results['query'], cur_results['pos'], cur_results['neg'], self.args.cl_temperature

        is_distributed = (self.args.process_index != -1)
        if is_distributed:
            gather_device_rank = 0 
            gathered_embeddings_query = [torch.zeros_like(embeddings_query) for _ in range(self.args.world_size)]
            dist.all_gather(gathered_embeddings_query, embeddings_query)
            gathered_embeddings_pos = [torch.zeros_like(embeddings_pos) for _ in range(self.args.world_size)]
            dist.all_gather(gathered_embeddings_pos, embeddings_pos)
            gathered_embeddings_neg = [torch.zeros_like(embeddings_neg) for _ in range(self.args.world_size)]
            dist.all_gather(gathered_embeddings_neg, embeddings_neg)

        def compute_constrastive_loss(embeddings_query, embeddings_pos, embeddings_neg, cl_temperature):
            num = len(embeddings_query)
            similarity_fct = nn.CosineSimilarity(dim=-1)

            # Compute similarity scores between query and pos/neg embeddings
            query_pos_sim = similarity_fct(embeddings_query, embeddings_pos) 
            query_neg_sims = similarity_fct(embeddings_query.unsqueeze(1), embeddings_neg)
            all_scores = torch.cat([query_pos_sim.unsqueeze(-1), query_neg_sims], dim=-1) 
            all_scores = all_scores / cl_temperature      
            labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
            contrastive_loss = nn.CrossEntropyLoss()(all_scores, labels)

            # Compute similarity scores between pos embeddings and query/neg embeddings
            all_another_scores = similarity_fct(embeddings_pos.unsqueeze(1), embeddings_query.unsqueeze(0))
            all_another_scores = all_another_scores / cl_temperature
            labels_another = torch.arange(0, num).long().to(embeddings_query.device)
            contrastive_loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)

            return contrastive_loss

        if not is_distributed:
            # <algo 0> 
            loss_tensor = compute_constrastive_loss(embeddings_query, embeddings_pos, embeddings_neg, cl_temperature) # simple DDP
            # </algo 0> 
        else:
            if self.args.process_index == gather_device_rank:
                # <algo 1> contacanate the embeddings from gpus and then compute loss and send them back to respective gpus
                gathered_embeddings_query = torch.cat(gathered_embeddings_query, dim=0).to(gather_device_rank)
                gathered_embeddings_query.requires_grad_()
                gathered_embeddings_pos = torch.cat(gathered_embeddings_pos, dim=0).to(gather_device_rank)
                gathered_embeddings_pos.requires_grad_()
                gathered_embeddings_neg = torch.cat(gathered_embeddings_neg, dim=0).to(gather_device_rank)
                gathered_embeddings_neg.requires_grad_()
                
                # Log the tensor size the first time
                if not hasattr(self, "printed_tensor_size"):
                    print(f"*** query embedding tensor size for loss: {gathered_embeddings_query.size()}")
                    print(f"*** pos embedding tensor size for loss: {gathered_embeddings_pos.size()}")
                    print(f"*** neg embedding tensor size for loss: {gathered_embeddings_neg.size()}")
                    self.printed_tensor_size = True

                loss = compute_constrastive_loss(gathered_embeddings_query, gathered_embeddings_pos, gathered_embeddings_neg, cl_temperature)
                scattered_loss_tensor = [torch.tensor([loss.item(),]).to(self.args.process_index) for _ in range(self.args.world_size)]
                
                loss.backward()

                scattered_embeddings_query_grad = list(gathered_embeddings_query.grad.split(embeddings_query.size(0)))
                scattered_embeddings_pos_grad = list(gathered_embeddings_pos.grad.split(embeddings_pos.size(0)))
                scattered_embeddings_neg_grad = list(gathered_embeddings_neg.grad.split(embeddings_neg.size(0)))
                # </algo 1> 

                # # <algo 2> compute loss for embeddings from each gpu separately and send them back to respective gpus
                # scattered_loss_tensor = []
                # scattered_embeddings_query_grad, scattered_embeddings_pos_grad, scattered_embeddings_neg_grad = [], [], []
                # for process_index, (cur_embeddings_query, cur_embeddings_pos, cur_embeddings_neg) in enumerate(zip(gathered_embeddings_query, gathered_embeddings_pos, gathered_embeddings_neg)):                    
                #     cur_embeddings_query.requires_grad_()
                #     cur_embeddings_pos.requires_grad_()
                #     cur_embeddings_neg.requires_grad_()

                #     cur_loss = compute_constrastive_loss(cur_embeddings_query, cur_embeddings_pos, cur_embeddings_neg, cl_temperature)
                #     scattered_loss_tensor.append(torch.tensor([cur_loss.item(),]).to(self.args.process_index))

                #     cur_loss.backward()

                #     scattered_embeddings_query_grad.append(cur_embeddings_query.grad)
                #     scattered_embeddings_pos_grad.append(cur_embeddings_pos.grad)
                #     scattered_embeddings_neg_grad.append(cur_embeddings_neg.grad)
                # # </algo 2> 
            else:
                scattered_embeddings_query_grad, scattered_embeddings_pos_grad, scattered_embeddings_neg_grad = None, None, None
                scattered_loss_tensor = None

            embeddings_query_grad = torch.ones_like(embeddings_query)
            dist.scatter(embeddings_query_grad, scattered_embeddings_query_grad, src=gather_device_rank)

            embeddings_pos_grad = torch.ones_like(embeddings_pos)
            dist.scatter(embeddings_pos_grad, scattered_embeddings_pos_grad, src=gather_device_rank)

            embeddings_neg_grad = torch.ones_like(embeddings_neg)
            dist.scatter(embeddings_neg_grad, scattered_embeddings_neg_grad, src=gather_device_rank)

            loss_tensor = torch.tensor([1.,]).to(self.args.process_index)
            dist.scatter(loss_tensor, scattered_loss_tensor, src=gather_device_rank)

            embeddings_query.backward(embeddings_query_grad) 
            embeddings_pos.backward(embeddings_pos_grad) 
            embeddings_neg.backward(embeddings_neg_grad) 

        return loss_tensor, is_distributed

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        with model.no_sync():
            model.train()
            inputs = self._prepare_inputs(inputs)

            if utils.is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss, is_distributed = self.compute_loss(model, inputs)

            if not is_distributed:
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                    # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                    loss = loss / self.args.gradient_accumulation_steps

                if self.do_grad_scaling:
                    self.scaler.scale(loss).backward()
                elif self.use_apex:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                elif self.deepspeed:
                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                    loss = self.deepspeed.backward(loss)
                else:
                    loss.backward()
            else:
                # grads computed inside the compute_loss function itself
                loss = loss[0]

        # # print some gradients for debugging across various algos
        # for rank in range(self.args.world_size):
        #     if self.args.process_index == rank:
        #         print(f"****** Rank {self.args.process_index}'s loss:", float(loss))
        #         for name, param in list(model.named_parameters())[-5:]:
        #             if param.grad is not None:
        #                 print(f"Rank {self.args.process_index}, Parameter {name}'s gradient hash: {tensor_info(param.grad)}") 
        #     dist.barrier()

        return loss.detach()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: str = field(default=None, metadata={"help": "Language id for summarization."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    processed_data_dir: Optional[str] = field(
        default=None, metadata={"help": "directory to the processed data"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    sample_selection_train_file_path: Optional[str] = field(
        default=None, metadata={"help": "sample_selection_train_file_path"}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    def_only: bool = field(
        default=False, metadata={"help": "def_only"}
    )
    add_prompt_to_document: bool = field(
        default=True, metadata={"help": "add_prompt_to_document"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    debug_mode: Optional[int] = field(
        default=None,
        metadata={"help": "debug mode"},
    )
    max_examples: Optional[int] = field(
        default=None,
        metadata={"help": "debug mode"},
    )
    cl_temperature: Optional[float] = field(
        default=None,
        metadata={"help": "temperature"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    sub_sample_ratio: Optional[float] = field(
        default=2.0,
        metadata={
            "help": (
                "sub_sample_ratio"
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    # local_rank: Optional[int] = field(default=-1, metadata={"help": "Local rank for distributed training"})
    def __post_init__(self):
        pass


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.output_dir = training_args.output_dir
    real_name_or_path = model_args.model_name_or_path
    data_args.model_name_or_path = model_args.model_name_or_path
    data_args.tokenizer_name_or_path = model_args.model_name_or_path
    training_args.cl_temperature = data_args.cl_temperature
    training_args.remove_unused_columns = False
    if not os.path.isdir(data_args.output_dir):
        os.makedirs(data_args.output_dir,exist_ok=True)

    # if data_args.local_rank != -1:
    #     print(f"Initializing distributed training with local rank {data_args.local_rank}")
    #     torch.cuda.set_device(data_args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.ERROR
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    set_seed(training_args.seed)
    with open(os.path.join(model_args.cache_dir, 'medi-data.json')) as f:
        train_examples_raw = json.load(f)
    if data_args.debug_mode:
        train_examples_raw = train_examples_raw[:data_args.debug_mode]
    old_train_examples_raw = train_examples_raw
    train_examples_raw = []
    total_n = len(old_train_examples_raw)
    real_batch_size = max(training_args.per_device_train_batch_size,
                          training_args.per_device_train_batch_size * torch.cuda.device_count())
    # print('real_batch_size: ', real_batch_size,training_args.per_device_train_batch_size,torch.cuda.device_count())
    for idx in range(0, total_n, real_batch_size):
        local_task_name = old_train_examples_raw[idx]['task_name']
        cur_batch = []
        include_batch = True
        for idx1 in range(idx, min(idx + real_batch_size, total_n)):
            if not old_train_examples_raw[idx1]['task_name'] == local_task_name:
                print(f'one batch in task {old_train_examples_raw[idx1]["task_name"]} is skipped')
                include_batch = False
                break
            else:
                cur_batch.append(old_train_examples_raw[idx1])
        if include_batch and len(cur_batch) == real_batch_size:
            train_examples_raw.append(cur_batch)
    random.shuffle(train_examples_raw)
    if data_args.max_examples is not None and len(train_examples_raw*real_batch_size)>data_args.max_examples:
        train_examples_raw = train_examples_raw[:int(data_args.max_examples/real_batch_size)]
    train_examples_raw_batch = train_examples_raw
    train_examples_raw = []
    for b in train_examples_raw_batch:
        train_examples_raw += b
    print(f'There are {len(train_examples_raw)} pairs to train in total')
    if data_args.debug_mode:
        train_examples_raw = train_examples_raw[:int(data_args.debug_mode)]

    train_examples = {'query':[],'pos':[],'neg':[],'task_name':[]}
    task_name_map = {}
    total_train_num = len(train_examples_raw)
    task_count = 0
    for i in range(total_train_num):
        cur_e = train_examples_raw[i]
        for k in ['query','pos','neg']:
            for s in cur_e[k][:-1]:
                assert not '!@#$%^&**!@#$%^&**' in s
            cur_e[k][-1] = str(cur_e[k][-1])
            if not data_args.add_prompt_to_document:
                cur_e[k][0] = ''
            assert cur_e[k][0].startswith('Represent ') or cur_e[k][0]==''
            train_examples[k].append('!@#$%^&**!@#$%^&**'.join(cur_e[k]))
        if not cur_e['task_name'] in task_name_map:
            task_name_map[cur_e['task_name']] = task_count
            task_count += 1
        train_examples['task_name'].append(task_name_map[cur_e['task_name']])
    raw_datasets = DatasetDict({'train':Dataset.from_dict(train_examples)})

    model = INSTRUCTOR(real_name_or_path, cache_folder=model_args.cache_dir)
    column_names = raw_datasets["train"].column_names

    def preprocess_function(examples):
        all_tokenized = None
        for key in ['query','pos','neg']:
            num = len(examples[key])
            contexts = []
            concatenated_input_texts = []
            for local_idx in range(num):
                splits = examples[key][local_idx].split('!@#$%^&**!@#$%^&**')
                assert len(splits) == 2
                contexts.append(splits[0])
                concatenated_input_texts.append(''.join(splits))
                assert isinstance(contexts[-1], str)
                assert isinstance(concatenated_input_texts[-1], str)
            tokenized = tokenizer(concatenated_input_texts,padding='max_length', truncation='longest_first', return_tensors="pt", max_length=data_args.max_source_length)
            context_tok = tokenizer(contexts,padding='max_length', truncation='longest_first', return_tensors="pt", max_length=data_args.max_source_length)
            tokenized['context_masks'] = torch.sum(context_tok['attention_mask'], dim=1)
            tokenized['context_masks'] = tokenized['context_masks'] - 1
            for my_idx in range(len(tokenized['context_masks'])):
                if tokenized['context_masks'][my_idx] <= 1:
                    tokenized['context_masks'][my_idx] = 0
            keys = tokenized.keys()
            if all_tokenized is None:
                all_tokenized = tokenized.copy()
                for k in keys:
                    all_tokenized[k] = all_tokenized[k].tolist()
            for k in keys:
                all_tokenized[f'{key}_{k}'] = tokenized[k].tolist()
        all_tokenized['task_name'] = examples['task_name']
        return all_tokenized

    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    trainer = InstructorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.model.save(training_args.output_dir)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
