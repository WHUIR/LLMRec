import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

from torch import autocast
from peft import LoraConfig, get_peft_model
from utils import print_rank0


class LLMRec(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        if 'bert' in args.backbone:
            from transformers import BertModel
            self.llm = BertModel.from_pretrained(args.root_path + args.backbone)
        elif 'opt' in args.backbone:
            from transformers import OPTModel
            self.llm = OPTModel.from_pretrained(args.root_path + args.backbone)
        elif 'flan' in args.backbone:
            from llm.modeling_t5 import T5EncoderModel
            self.llm = T5EncoderModel.from_pretrained(args.root_path + args.backbone)

        if args.lora:
            # if args.debug or not args.pretrain_lora:
            print_rank0("Initialize Lora From Scratch!", self.args.rank)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
            )
            self.llm = get_peft_model(self.llm, config)
            self.trainable2float()

        self.item_embs = None

    # 将可学习的参数都转换成float32，不然amp会出问题
    def trainable2float(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print_rank0(f"Trainable Parameter:{name}", self.args.rank)
                param.data = param.data.float()

    def get_embedding(self, input_ids, attention_mask):
        llm_output = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        if 'bert' == self.args.backbone[:4]:
            return llm_output[0][:, 0]
        elif 'opt' == self.args.backbone[:3]:
            return self.gather_indexes(llm_output.last_hidden_state, attention_mask.sum(dim=-1) - 1)
        elif 'flan' == self.args.backbone[:4]:
            return self.gather_indexes(llm_output.last_hidden_state, attention_mask.sum(dim=-1) - 1)

    def forward(self, inputs):
        device = next(self.parameters()).device
        seq_cls = self.get_embedding(input_ids=inputs['sequence_input_ids'], attention_mask=inputs['sequence_attention_mask'])
        item_cls = self.get_embedding(input_ids=inputs['item_input_ids'], attention_mask=inputs['item_attention_mask'])
        item_cls = item_cls.view(seq_cls.size()[0], self.args.train_nega_count + 1, item_cls.size()[-1])

        with autocast(device_type='cuda', enabled=False):
            item_cls = item_cls.float()
            seq_cls = seq_cls.float().unsqueeze(-1)
            scores = torch.bmm(item_cls, seq_cls).squeeze(-1)
            loss = F.cross_entropy(scores, inputs['target_position'])
        return [loss, loss]

    def valid_step(self, inputs):
        seq_cls = self.get_embedding(input_ids=inputs['sequence_input_ids'], attention_mask=inputs['sequence_attention_mask'])
        item_cls = self.item_embs[inputs['negative_items']].to(seq_cls.device)

        with autocast(device_type='cuda', enabled=False):
            item_cls = item_cls.float()
            seq_cls = seq_cls.float().unsqueeze(-1)
            scores = torch.bmm(item_cls, seq_cls).squeeze(-1) / math.sqrt(item_cls.size()[-1])
            loss = F.cross_entropy(scores, inputs['target_position'])

        return scores, inputs['target_position']

    @torch.no_grad()
    def generate_embs(self, item_tokens):
        del self.item_embs
        torch.cuda.empty_cache()
        print_rank0(f"GPU:{self.args.rank} Generating Emebedding")
        item_ids = item_tokens['item_ids']
        item_attn = item_tokens['item_attn']
        device = next(self.parameters()).device

        item_embs = []
        batch_size = 128
        if self.args.rank == 0:
            iterator = tqdm(range(0, item_ids.size()[0], batch_size), desc='Generate embs')
        else:
            iterator = range(0, item_ids.size()[0], batch_size)
        for start_idx in iterator:
            batch_item_ids = item_ids[start_idx: start_idx + batch_size].to(device)
            batch_item_attn = item_attn[start_idx: start_idx + batch_size].to(device)
            batch_item_embs = self.get_embedding(input_ids=batch_item_ids, attention_mask=batch_item_attn)
            item_embs.append(batch_item_embs.detach())
        self.item_embs = torch.cat(item_embs, dim=0)
        assert self.item_embs.size()[0] == item_ids.size()[0]

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
