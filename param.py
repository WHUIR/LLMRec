import argparse
import random

import numpy as np
import torch

import pprint
import yaml


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')


def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # Data Splits
    parser.add_argument("--dataset", type=str, default='mSPIAO')
    parser.add_argument("--item_count", type=int, default=0)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--user_k', type=int, default=5)
    parser.add_argument('--item_k', type=int, default=5)
    parser.add_argument('--ratio', type=float, default=1.0)
    parser.add_argument('--nega_count', type=int, default=1000)
    parser.add_argument('--train_nega_count', type=int, default=10)


    # Checkpoint
    parser.add_argument('--output', type=str, default='./ckp/{}.pth')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--save_by_step', type=int, default=3000, help='save model by step or epoch')

    # CPU/GPU
    # parser.add_argument("--multdiGPU", action='store_const', default=False, const=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--multiGPU", action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--num_gpus", default=4, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--port', type=int, default=12347)
    parser.add_argument('--valid_ratio', type=float, default=1.0)

    # Model Config
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--valid_first', action='store_true')
    parser.add_argument('--root_path', type=str, default='plm_models')
    parser.add_argument('--backbone', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_token_length', type=int, default=1024)
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--num_works', type=int, default=4)
    parser.add_argument('--item_emb_dim', type=int, default=128)

    # Lora Config
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', type=list, default=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    parser.add_argument('--pretrain_lora', action='store_true')


    # Training
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--optim', default='adamw')         
    parser.add_argument('--warmup_ratio', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--adam_eps', type=float, default=1e-4)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--skip_valid', action='store_true')
    parser.add_argument('--use_cache', action='store_false')
    parser.add_argument('--start_epoch', type=int, default=0)

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 配置args
    args.fp16 = True
    args.valid_first = True
    args.distributed = True
    args.multiGPU = True
    args.valid_ratio = 0.1

    import math
    args.gradient_accumulation_steps = math.ceil(96 / args.batch_size / args.num_gpus)

    return args


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


if __name__ == '__main__':
    args = parse_args(True)
