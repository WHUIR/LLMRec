from pathlib import Path
import time
import torch
import torch.nn as nn
from pprint import pprint

from utils import load_state_dict, LossMeter
from pprint import pformat

proj_dir = Path(__file__).resolve().parent.parent


class TrainerBase(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = None

        self.verbose = True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False


    def create_config(self):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.args.root_path + self.args.backbone)
        args = self.args
        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout
        config.losses = args.losses
        return config


    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None

        if 'adamw' in self.args.optim:
            from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

            batch_per_epoch = len(self.train_loader)
            t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epoch
            warmup_ratio = self.args.warmup_ratio
            warmup_iters = int(t_total * warmup_ratio)

            if self.verbose:
                print("Batch per epoch: %d" % batch_per_epoch)
                print("Total Iters: %d" % t_total)
                print('Warmup ratio:', warmup_ratio)
                print("Warm up Iters: %d" % warmup_iters)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)
            lr_scheduler = get_linear_schedule_with_warmup(
                optim, warmup_iters, t_total)
            # lr_scheduler = get_constant_schedule(optim)

        else:
            optim = self.args.optimizer(
                list(self.model.parameters()), self.args.lr)

        return optim, lr_scheduler

    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')
        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)

    def init_weights(self):

        def init_bert_weights(module):
            """ Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=1)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.model.apply(init_bert_weights)
        self.model.init_weights()

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save(self, path):
        if self.args.k8s:
            from alps.pytorch.api.utils.save_load_util import alps_torch_save, alps_torch_load
            alps_torch_save(self.model.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def load(self, path, loc=None):
        if loc is None and hasattr(self.args, 'gpu'):
            loc = f'cuda:{self.args.gpu}'
        state_dict = torch.load("%s.pth" % path, map_location=loc)
        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', path)
            pprint(results)

    def remain_time(self, epoch_start_time, step_i, loader_length):
        remain_time = (time.time() - epoch_start_time) * (loader_length - step_i - 1) / (step_i + 1)
        remain_hour = remain_time // 3600
        remain_min = (remain_time-remain_hour*3600) // 60
        remain_sec = round(remain_time-remain_hour*3600-remain_min*60, 4)
        return int(remain_hour), int(remain_min), remain_sec
