from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from tqdm import tqdm
import os
import numpy as np
import random
import time
from utils import print_rank0

class DataSequential(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super().__init__()
        print_rank0(f"Loading {mode} data...", args.rank)
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode
        self.length = 0
        self.data = None
        self.max_seq_length = 10
        self.max_token_length = args.max_token_length
        self.item_title_list = None
        self.nega_items = None
        self.item_count = max(list(pickle.load(open(f"{args.data_path}/{args.dataset}/iid2asin.pkl", 'rb')).keys())) + 1
        self.single_domain_iid = pickle.load(open(f'{args.data_path}/{args.dataset}/single_domain_iid.pkl', 'rb'))

        self.load_data()
        self.item_title_tokens = None

        self.tokenize_item_titles()
        self.load_negative()
        self.sample_valid(self.data)


        self.candi_item_attention_mask = None
        self.candi_item_input_ids = None
        self.generate_cate_items()
        print_rank0(f"Load {mode} data successfully", args.rank)


    def sample_valid(self, datas):
        if self.args.valid_ratio == 1 or self.mode != 'valid':
            return
        import random
        random.seed(42)
        sample_idx = random.sample(list(range(len(datas))), int(len(datas) * self.args.valid_ratio))
        sample_idx.sort()
        new_datas = []
        for idx in sample_idx:
            new_datas.append(datas[idx])
        self.nega_items = self.nega_items[sample_idx]
        self.length = len(new_datas)
        self.data = new_datas

    def load_negative(self):
        if self.mode == 'train':
            print_rank0("Don't load negatives!", self.args.rank)
            return
        self.nega_items = pickle.load(open(f'{self.args.data_path}/{self.args.dataset}/negatives_{self.mode}-{self.args.nega_count}.pkl', 'rb'))
        print_rank0("Load negatives successfully", self.args.rank)


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        example_input = self.generate_example_input(self.data[item], item)
        example_input.append(item)
        return example_input

    def load_data(self):
        if os.path.exists(f'local_dataset/{self.args.dataset}/train_data.pkl'):
            if self.mode == 'train':
                self.data = pickle.load(open(f'local_dataset/{self.args.dataset}/train_data.pkl', 'rb'))
            elif self.mode == 'valid':
                self.data = pickle.load(open(f'local_dataset/{self.args.dataset}/valid_data.pkl', 'rb'))
            elif self.mode == 'test':
                self.data = pickle.load(open(f'local_dataset/{self.args.dataset}/test_data.pkl', 'rb'))
            self.length = len(self.data)
            return

        review_datas = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/review_datas.pkl", 'rb'))
        train_data = []
        valid_data = []
        test_data = []

        for user in tqdm(review_datas.keys(), desc='Splitting Train/Valid/Test'):
            seq_iid_list = [review_datas[user][0][0]]
            seq_iid_cate_list = [review_datas[user][0][2]]
            for i in range(1, len(review_datas[user])):
                target_iid = review_datas[user][i][0]
                target_iid_cate = review_datas[user][i][2]
                if i < len(review_datas[user]) - 2:
                    train_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                elif i == len(review_datas[user]) - 2:
                    valid_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                elif i == len(review_datas[user]) - 1:
                    test_data.append([seq_iid_list, target_iid, seq_iid_cate_list, target_iid_cate])
                else:
                    raise NotImplementedError
                seq_iid_list = seq_iid_list + [review_datas[user][i][0]]
                seq_iid_cate_list = seq_iid_cate_list + [review_datas[user][i][2]]

                seq_iid_list = seq_iid_list[-self.max_seq_length:]
                seq_iid_cate_list = seq_iid_cate_list[-self.max_seq_length:]

        if self.args.rank == 0:
            os.makedirs(f'local_dataset/{self.args.dataset}/', exist_ok=True)
            pickle.dump(train_data, open(f'local_dataset/{self.args.dataset}/train_data.pkl', 'wb'))
            pickle.dump(valid_data, open(f'local_dataset/{self.args.dataset}/valid_data.pkl', 'wb'))
            pickle.dump(test_data, open(f'local_dataset/{self.args.dataset}/test_data.pkl', 'wb'))
        else:
            time.sleep(20)

        if self.mode == 'train':
            self.data = train_data
        elif self.mode == 'valid':
            self.data = valid_data
        elif self.mode == 'test':
            self.data = test_data
        else:
            raise NotImplementedError
        self.length = len(self.data)

    def generate_cate_items(self):
        candi_item_input_ids = []
        candi_item_attention_mask = []
        fp_tokens = 42
        for idx in range(self.item_count):
            if 'bert' in self.args.backbone:
                candi_tokens = [self.tokenizer.cls_token_id] + self.item_title_tokens[idx]
            elif 'opt' in self.args.backbone:
                candi_tokens = self.item_title_tokens[idx] + [self.tokenizer.eos_token_id]
            elif 'flan' in self.args.backbone:
                candi_tokens = self.item_title_tokens[idx] + [self.tokenizer.eos_token_id]
            else:
                raise NotImplementedError
            pad_len = fp_tokens - len(candi_tokens)
            candi_item_input_ids.append(candi_tokens + [0] * pad_len)
            candi_item_attention_mask.append((len(candi_tokens) * [1] + [0] * pad_len))
        self.candi_item_input_ids = candi_item_input_ids
        self.candi_item_attention_mask = candi_item_attention_mask

    def tokenize_item_titles(self):
        if os.path.exists(f'local_dataset/{self.args.dataset}/tokenized_{self.args.backbone}.pkl'):
            tokenized = pickle.load(open(f'local_dataset/{self.args.dataset}/tokenized_{self.args.backbone}.pkl', 'rb'))
            self.item_title_tokens = tokenized['item_title_tokens']
            return

        item_metas = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/meta_datas.pkl", 'rb'))
        iid2asin = pickle.load(open(f"{self.args.data_path}/{self.args.dataset}/iid2asin.pkl", 'rb'))
        item_title_list = ['None'] * self.item_count
        for iid, asin in iid2asin.items():
            item_title = item_metas[asin]['title'] if (
                    'title' in item_metas[asin].keys() and item_metas[asin]['title']) else 'None'
            item_title = item_title + '; '
            item_title_list[iid] = item_title


        item_max_tokens = 40
        item_title_tokens = []
        for start in tqdm(range(0, len(item_title_list), 32), desc='Tokenizing'):
            tokenized_text = self.tokenizer(item_title_list[start: start + 32],
                                            truncation=True,
                                            max_length=item_max_tokens,
                                            padding=False,
                                            add_special_tokens=False,
                                            return_tensors=None)

            item_title_tokens.extend(tokenized_text['input_ids'])

        self.item_title_tokens = item_title_tokens


        tokenized = {
            'item_title_tokens': self.item_title_tokens,
        }
        if self.args.rank == 0:
            pickle.dump(tokenized, open(f'local_dataset/{self.args.dataset}/tokenized_{self.args.backbone}.pkl', 'wb'))
        else:
            time.sleep(10)

    def generate_example_input(self, example, example_idx):
        seq_iid_list, target_iid = example[0], example[1]

        sequence_input_ids = []
        sequence_attention_mask = []  # 每个序列都会有一个

        fp_tokens = 42

        for seq_iid in seq_iid_list:
            sequence_attention_mask.extend([1] * len(self.item_title_tokens[seq_iid]))
            sequence_input_ids.extend(self.item_title_tokens[seq_iid])

        if 'bert' == self.args.backbone[: 4]:
            sequence_input_ids = [self.tokenizer.cls_token_id] + sequence_input_ids
            sequence_attention_mask.append(1)
        elif 'opt' == self.args.backbone[: 3]:
            sequence_input_ids = sequence_input_ids + [self.tokenizer.eos_token_id]
            sequence_attention_mask.append(1)
        elif 'flan' == self.args.backbone[: 4]:
            sequence_input_ids = sequence_input_ids + [self.tokenizer.eos_token_id]
            sequence_attention_mask.append(1)

        if self.mode == 'train':
            # negative_items = np.random.randint(1, self.item_count, size=self.args.nega_count).tolist()
            negative_items = random.sample(self.single_domain_iid[example[3]], self.args.train_nega_count)
            target_position = random.randint(0, self.args.train_nega_count)
        else:
            negative_items = self.nega_items[example_idx].tolist()
            target_position = random.randint(0, self.args.nega_count)
        # target_position = random.randint(0, self.args.nega_count - 1)
        negative_items = negative_items[0:target_position] + [target_iid] + negative_items[target_position:]

        if self.mode == 'train':
            candi_item_input_ids = [self.candi_item_input_ids[x] for x in negative_items]
            candi_item_attention_mask = [self.candi_item_attention_mask[x] for x in negative_items]
        else:
            candi_item_input_ids = [0] * len(negative_items)
            candi_item_attention_mask = [0] * len(negative_items)

        return [candi_item_input_ids, candi_item_attention_mask, sequence_attention_mask, sequence_input_ids, target_position, target_iid, negative_items]


    def collate_fn(self, batch_data):
        # candi_item_input_ids, candi_item_attention_mask, sequence_attention_mask, sequence_input_ids, target_position, target_iid, negative_items

        item_input_ids = []
        item_attention_mask = []
        sequence_attention_mask = []
        sequence_input_ids = []
        target_position = []
        target_iid = []
        example_index = []
        negative_items = []

        max_seq_length = max(len(x[2]) for x in batch_data)

        for example in batch_data:
            item_input_ids.extend(example[0])
            item_attention_mask.extend(example[1])

            seq_pad_len = max_seq_length - len(example[2])
            sequence_attention_mask.append(example[2] + seq_pad_len * [0])
            sequence_input_ids.append(example[3] + seq_pad_len * [0])
            target_position.append(example[4])
            target_iid.append(example[5])
            negative_items.append(example[6])
            example_index.append(example[-1])

        return {
            'item_input_ids': torch.LongTensor(item_input_ids),
            'item_attention_mask': torch.LongTensor(item_attention_mask),
            'sequence_attention_mask': torch.LongTensor(sequence_attention_mask),
            'sequence_input_ids': torch.LongTensor(sequence_input_ids),
            'target_position': torch.LongTensor(target_position),
            'target_iid': torch.LongTensor(target_iid),
            'example_index': torch.LongTensor(example_index),
            'negative_items': torch.LongTensor(negative_items)
        }

    def get_items_tokens(self):
        item_ids = []
        item_attn = []
        fp_tokens = 42
        for iid in range(len(self.item_title_tokens)):
            if 'bert' == self.args.backbone[: 4]:
                item_tokens = [self.tokenizer.cls_token_id] + self.item_title_tokens[iid]
            elif 'opt' == self.args.backbone[: 3]:
                item_tokens = self.item_title_tokens[iid] + [self.tokenizer.eos_token_id]
            elif 'flan' == self.args.backbone[: 4]:
                item_tokens = self.item_title_tokens[iid] + [self.tokenizer.eos_token_id]
            else:
                raise NotImplementedError
            pad_len = fp_tokens - len(item_tokens)
            item_ids.append(item_tokens + [0] * pad_len)
            item_attn.append(len(item_tokens) * [1] + pad_len * [0])
        return {'item_ids': torch.LongTensor(item_ids),
                'item_attn': torch.LongTensor(item_attn)}