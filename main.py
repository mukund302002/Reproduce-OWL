import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaTokenizer
# from importlib.metadata import version
from collections import defaultdict
from lib.prune_all import prune_wanda_outlier_structure_special,prune_wanda_outlier_structure,prune_sparsegpt_outlier,prune_wanda_outlier,prune_mag_outlier, prune_wanda,prune_magnitude,prune_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl
import sys
print('# of gpus: ', torch.cuda.device_count())



import json
import logging
import math
from datetime import datetime, timezone

import random
from itertools import chain
from pathlib import Path

import datasets

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = 2048
    return model

def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise AttributeError("Could not locate transformer layers on this model")


## Mukund
def compute_layerwise_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layerwise = []
    layers = get_transformer_layers(model)

    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        zero_params = 0
        total_params = 0
        for name in subset:
            W = subset[name].weight.data
            zero_params += (W == 0).sum().item()
            total_params += W.numel()

        layer_sparsity = float(zero_params) / float(total_params) if total_params > 0 else 0.0
        layerwise.append(
            {
                "layer_index": i,
                "sparsity_ratio": layer_sparsity,
                "zero_params": int(zero_params),
                "total_params": int(total_params),
            }
        )

    model.config.use_cache = use_cache
    return layerwise


def compute_layerwise_submodule_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layerwise = []
    layers = get_transformer_layers(model)
    for i, layer in enumerate(layers):
        subset = find_layers(layer)
        submodules = []
        for name in sorted(subset.keys()):
            W = subset[name].weight.data
            zero_params = int((W == 0).sum().item())
            total_params = int(W.numel())
            sparsity_ratio = float(zero_params) / float(total_params) if total_params > 0 else 0.0
            submodules.append(
                {
                    "submodule": name,
                    "sparsity_ratio": sparsity_ratio,
                    "zero_params": zero_params,
                    "total_params": total_params,
                    "pruned_percent": sparsity_ratio * 100.0,
                }
            )
        layerwise.append(
            {
                "layer_index": i,
                "submodules": submodules,
            }
        )

    model.config.use_cache = use_cache
    return layerwise
## MUKUND

def main():


    ########################## for prune ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str)
    parser.add_argument("--prune_method", type=str)
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')


########################################### for train
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="wikitext-2-raw-v1",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )



   

    #### saving parameters #####
    
    parser.add_argument(
        "--method",
        type=str,
        default=None,

    )   
    


    #### data parameters #####
    
    parser.add_argument(
        "--Lamda",
        default=0.08,
        type=float,
        help="Lamda",
    )
    
    
     
    parser.add_argument(
        '--Hyper_m', 
        type=float,
        default=3, )
    
    parser.add_argument(
    "--outlier_by_activation", action="store_true", help="outlier_by_activation")  
    
    
    parser.add_argument(
    "--outlier_by_wmetric", action="store_true", help="outlier_by_wmetric")  
   
        
    
    args = parser.parse_args()


    print ("args.nsamples",args.nsamples)
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))


    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    
    
    print ("model is =================================================================================")
    print (model.__class__.__name__)
    print (model)
    
    
    model.eval()
    
    if "opt" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    elif "llama" in args.model:
    
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)



    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)



    print ("target sparsity", args.sparsity_ratio)   







    print("pruning starts")


    ############################ baseline   ############################
    if args.prune_method == "wanda":


        prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)




    elif args.prune_method == "magnitude":


        prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)





    elif args.prune_method == "sparsegpt":


        prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    ############################ owl   ############################
    elif args.prune_method == "wanda_owl":

        prune_wanda_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    ############################ owl   ############################
    elif args.prune_method == "magnitude_owl":

        prune_mag_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)




    elif args.prune_method == "sparsegpt_owl":
    
        prune_sparsegpt_outlier(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


    elif args.prune_method == "wanda_owl_structure":


        prune_wanda_outlier_structure(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        
        
    elif args.prune_method == "wanda_owl_structure_special":
        prune_wanda_outlier_structure_special(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)


         
    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl = eval_ppl(model, tokenizer, device)
    print(f"ppl on wikitext {ppl}")
    layerwise_sparsity = compute_layerwise_sparsity(model)
    layerwise_submodule_sparsity = compute_layerwise_submodule_sparsity(model)

    sys.stdout.flush()

    # print(f"final ppl on wikitext {ppl}")


## MUKUND
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        result_entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "prune_method": args.prune_method,
            "sparsity_type": args.sparsity_type,
            "target_sparsity_ratio": args.sparsity_ratio,
            "actual_sparsity_ratio": float(sparsity_ratio),
            "layerwise_sparsity": layerwise_sparsity,
            "layerwise_submodule_sparsity": layerwise_submodule_sparsity,
            "nsamples": args.nsamples,
            "seed": args.seed,
            "ppl_wikitext": float(ppl),
            "Lamda": args.Lamda,
            "Hyper_m": args.Hyper_m,
            "outlier_by_activation": bool(args.outlier_by_activation),
            "outlier_by_wmetric": bool(args.outlier_by_wmetric),
        }

        existing_results = []
        if save_path.exists() and save_path.stat().st_size > 0:
            try:
                with save_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    existing_results = loaded
                elif isinstance(loaded, dict):
                    existing_results = [loaded]
            except json.JSONDecodeError:
                print(f"warning: {save_path} contains invalid JSON, overwriting with a new results list")

        existing_results.append(result_entry)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=2)
        print(f"results saved to {save_path}")
## MUKUND

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"model saved to {args.save_model}")








if __name__ == '__main__':
    main()
