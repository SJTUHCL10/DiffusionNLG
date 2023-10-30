import datasets
import torch
import os
import pickle
import random

from collections import Counter
from spacy.lang.en import English
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertTokenizerFast


def load_e2enlg_dataset_and_tokenizer(max_len=64, vocab_threshold=10):
    """
    load e2e_nlg dataset using huggingface datasets and create vocab dictionary
    :param max_len: maximum length of one sentence
    :param vocab_threshold: minimum occurrence time of tokens in vocab dict
    :return: tokenized dataset and vocab dictionary (tokenizer)
    """

    e2enlg_dataset = datasets.load_dataset("e2e_nlg")
    nlp = English()
    tokenizer = nlp.tokenizer
    sentence_lst = []
    for sentence in tqdm(e2enlg_dataset["train"]["human_reference"]):
        word_lst = [x.text for x in tokenizer(sentence)]
        sentence_lst.append(word_lst)

    # build vocab dict
    counter = Counter()
    for tokenized_sentence in sentence_lst:
        counter.update(tokenized_sentence)

    vocab_dict = {'[START]': 0, '[END]': 1, '[UNK]': 2, '[PAD]': 3}
    for k, v in counter.items():
        if v >= vocab_threshold:
            vocab_dict[k] = len(vocab_dict)

    # print(len(counter), len(vocab_dict))

    raw_e2enlg_dataset = datasets.Dataset.from_dict({"text": sentence_lst})

    # tokenize and pad
    def tokenize_func(samples):
        input_ids = []
        attention_mask = []
        for seq in samples['text']:
            if len(seq) < max_len-1:
                input_ids.append([0] + [vocab_dict.get(x, vocab_dict['[UNK]']) for x in seq] + [1] + [vocab_dict['[PAD]']]*(max_len-len(seq)-2))
                attention_mask.append([1]*(len(seq)+2) + [0]*(max_len-len(seq)-2))
            else:
                input_ids.append([0] + [vocab_dict.get(x, vocab_dict['[UNK]']) for x in seq[:max_len-2]] + [1])
                attention_mask.append([1]*max_len)
        result_dict = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return result_dict

    tokenized_e2enlg_dataset = raw_e2enlg_dataset.map(tokenize_func, batched=True)

    return tokenized_e2enlg_dataset, vocab_dict


class E2enlgDataset(Dataset):
    def __init__(self, data_lst, attention_mask_lst):
        self.data_tensor = torch.LongTensor(data_lst)   # [dataset_size, seq_len]
        self.attention_mask_tensor = torch.Tensor(attention_mask_lst)

    def __getitem__(self, idx):
        input_ids = self.data_tensor[idx, :]
        attention_mask = self.attention_mask_tensor[idx, :]
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def __len__(self):
        return self.data_tensor.shape[0]


def load_rocstories_dataset_and_tokenizer(max_len=72, vocab_threshold=10):
    rocstories_dataset = datasets.load_dataset("wza/roc_stories")
    nlp = English()
    tokenizer = nlp.tokenizer
    sentence_lst = []

    for sample in tqdm(rocstories_dataset['train']):
        sentence = sample['sentence1'] + ' ' + sample['sentence2'] + ' ' + sample['sentence3'] + ' ' + sample['sentence4'] + ' ' + sample['sentence5']
        word_lst = [x.text for x in tokenizer(sentence)]
        sentence_lst.append(word_lst)

    counter = Counter()
    for tokenized_sentence in sentence_lst:
        counter.update(tokenized_sentence)

    vocab_dict = {'[START]': 0, '[END]': 1, '[UNK]': 2, '[PAD]': 3}
    for k, v in counter.items():
        if v >= vocab_threshold:
            vocab_dict[k] = len(vocab_dict)

    raw_rocstories_dataset = datasets.Dataset.from_dict({"text": sentence_lst})

    # tokenize and pad
    def tokenize_func(samples):
        input_ids = []
        attention_mask = []
        for seq in samples['text']:
            if len(seq) < max_len-1:
                input_ids.append([0] + [vocab_dict.get(x, vocab_dict['[UNK]']) for x in seq] + [1] + [vocab_dict['[PAD]']]*(max_len-len(seq)-2))
                attention_mask.append([1]*(len(seq)+2) + [0]*(max_len-len(seq)-2))
            else:
                input_ids.append([0] + [vocab_dict.get(x, vocab_dict['[UNK]']) for x in seq[:max_len-2]] + [1])
                attention_mask.append([1]*max_len)
        result_dict = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return result_dict

    tokenized_rocstories_dataset = raw_rocstories_dataset.map(tokenize_func, batched=True)

    return tokenized_rocstories_dataset, vocab_dict


class RocstoriesDataset(Dataset):
    def __init__(self, data_lst, attention_mask_lst):
        self.data_tensor = torch.LongTensor(data_lst)   # [dataset_size, seq_len]
        self.attention_mask_tensor = torch.Tensor(attention_mask_lst)

    def __getitem__(self, idx):
        input_ids = self.data_tensor[idx, :]
        attention_mask = self.attention_mask_tensor[idx, :]
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def __len__(self):
        return self.data_tensor.shape[0]


def build_qqp_dataset_and_tokenizer(max_len=32,
                                    vocab_threshold=5,
                                    bert_tokenzier_name='bert-base-uncased',
                                    save_dir=None):
    # qqp_dataset = datasets.load_dataset("glue", "qqp")
    qqp_dataset = datasets.load_from_disk("data/qqp")
    qqp_paraphrase_dataset = qqp_dataset.filter(lambda example: example['label'] == 1)

    nlp = English()
    tokenizer = nlp.tokenizer
    bert_tokenizer = BertTokenizerFast.from_pretrained(bert_tokenzier_name)

    # build custom tokenizer
    total_sentence_lst = []
    train_sentence_lst1, train_sentence_lst2, eval_sentence_lst1, eval_sentence_lst2 = [], [], [], []
    for sentence in tqdm(qqp_paraphrase_dataset["train"]["question1"]):
        word_lst = [x.text for x in tokenizer(sentence)]
        train_sentence_lst1.append(word_lst)
        total_sentence_lst.append(word_lst)

    for sentence in tqdm(qqp_paraphrase_dataset["train"]["question2"]):
        word_lst = [x.text for x in tokenizer(sentence)]
        train_sentence_lst2.append(word_lst)
        total_sentence_lst.append(word_lst)

    for sentence in tqdm(qqp_paraphrase_dataset["validation"]["question1"]):
        word_lst = [x.text for x in tokenizer(sentence)]
        eval_sentence_lst1.append(word_lst)
        total_sentence_lst.append(word_lst)

    for sentence in tqdm(qqp_paraphrase_dataset["validation"]["question2"]):
        word_lst = [x.text for x in tokenizer(sentence)]
        eval_sentence_lst2.append(word_lst)
        total_sentence_lst.append(word_lst)

    counter = Counter()
    for tokenized_sentence in total_sentence_lst:
        counter.update(tokenized_sentence)

    vocab_dict = {'[START]': 0, '[END]': 1, '[UNK]': 2, '[PAD]': 3}
    for k, v in counter.items():
        if v >= vocab_threshold:
            vocab_dict[k] = len(vocab_dict)

    # print(len(counter), len(vocab_dict))
    raw_train_qqp = datasets.Dataset.from_dict(
        {"question1_text": qqp_paraphrase_dataset['train']['question1'],
         "question2_text": qqp_paraphrase_dataset['train']['question2'],
         "question1_wordlst": train_sentence_lst1,
         "question2_wordlst": train_sentence_lst2})
    raw_eval_qqp = datasets.Dataset.from_dict(
        {"question1_text": qqp_paraphrase_dataset['validation']['question1'],
         "question2_text": qqp_paraphrase_dataset['validation']['question2'],
         "question1_wordlst": eval_sentence_lst1,
         "question2_wordlst": eval_sentence_lst2})

    def total_tokenize_func(samples):
        res = {}

        # using bert tokenizer
        # input_ids, token_type_ids, attention_masks
        tmp = bert_tokenizer(samples["question1_text"], padding='max_length', truncation=True)
        for k, v in tmp.items():
            res['question1_' + k + '_bert'] = v
        tmp = bert_tokenizer(samples["question2_text"], padding='max_length', truncation=True)
        for k, v in tmp.items():
            res['question2_' + k + '_bert'] = v

        # using custom tokenizer
        question1_input_ids, question2_input_ids = [], []
        question1_attention_mask, question2_attention_mask = [], []
        for seq in samples['question1_wordlst']:
            if len(seq) < max_len-1:
                question1_input_ids.append([0] + [vocab_dict.get(x, vocab_dict['[UNK]']) for x in seq] + [1] + [vocab_dict['[PAD]']]*(max_len-len(seq)-2))
                question1_attention_mask.append([1]*(len(seq)+2) + [0]*(max_len-len(seq)-2))
            else:
                question1_input_ids.append([0] + [vocab_dict.get(x, vocab_dict['[UNK]']) for x in seq[:max_len-2]] + [1])
                question1_attention_mask.append([1]*max_len)

        for seq in samples['question2_wordlst']:
            if len(seq) < max_len-1:
                question2_input_ids.append([0] + [vocab_dict.get(x, vocab_dict['[UNK]']) for x in seq] + [1] + [vocab_dict['[PAD]']]*(max_len-len(seq)-2))
                question2_attention_mask.append([1]*(len(seq)+2) + [0]*(max_len-len(seq)-2))
            else:
                question2_input_ids.append([0] + [vocab_dict.get(x, vocab_dict['[UNK]']) for x in seq[:max_len-2]] + [1])
                question2_attention_mask.append([1]*max_len)

        res['question1_input_ids'] = question1_input_ids
        res['question2_input_ids'] = question2_input_ids
        res['question1_attention_mask'] = question1_attention_mask
        res['question2_attention_mask'] = question2_attention_mask

        return res

    qqp_train = raw_train_qqp.map(total_tokenize_func, batched=True)
    qqp_eval = raw_eval_qqp.map(total_tokenize_func, batched=True)

    if save_dir is not None:
        print("saving dataset and vocab_dict to ", save_dir)
        qqp_train.save_to_disk(os.path.join(save_dir, "qqp_paraphrase_train"))
        qqp_eval.save_to_disk(os.path.join(save_dir, "qqp_paraphrase_eval"))
        with open(os.path.join(save_dir, "qqp_paraphrase_vocab_dict"), "wb") as f:
            pickle.dump(vocab_dict, f)

    return qqp_train, qqp_eval, vocab_dict


def load_qqp_dataset_and_tokenizer_from_disk(data_path="data", test_size=0.1):

    qqp_train = datasets.load_from_disk(os.path.join(data_path, "qqp_paraphrase_train"))
    # qqp_eval = datasets.load_from_disk(os.path.join(data_path, "qqp_paraphrase_eval"))
    qqp_train, qqp_eval = train_test_split(qqp_train, test_size=test_size, shuffle=True)    # split training set into train and dev
    with open(os.path.join(data_path, "qqp_paraphrase_vocab_dict"), "rb") as f:
        vocab_dict = pickle.load(f)

    return qqp_train, qqp_eval, vocab_dict


class QQPParaphraseDataset(Dataset):
    def __init__(self, dataset, random_swap=False):
        """
        QQP paraphrase dataset using torch.utils.data.Dataset
        :param dataset: huggingface dataset
        :param random_swap: randomly swap question1 and question2
        """
        longtensor_keys = ['question1_input_ids_bert',
                           'question2_input_ids_bert',
                           'question1_input_ids',
                           'question2_input_ids']
        tensor_keys = ['question1_attention_mask_bert',
                       'question2_attention_mask_bert',
                       'question1_attention_mask',
                       'question2_attention_mask']

        dict_dataset = {}
        for key in longtensor_keys:
            dict_dataset[key] = torch.LongTensor(dataset[key])
        for key in tensor_keys:
            dict_dataset[key] = torch.Tensor(dataset[key])
        self.dataset = dict_dataset
        self.random_swap = random_swap

    def flip_key(self, key_str):
        # used to swap 1 and 2
        # ord('1') + ord('2') = 99
        return key_str[:8] + chr(99-ord(key_str[8])) + key_str[9:]

    def __getitem__(self, item):
        return_keys = ['question1_input_ids_bert',
                       'question1_attention_mask_bert',
                       'question2_input_ids_bert',
                       'question2_attention_mask_bert',
                       'question1_input_ids',
                       'question2_input_ids',
                       'question1_attention_mask',
                       'question2_attention_mask']

        if self.random_swap and random.random() < 0.5:
            return {k: self.dataset[self.flip_key(k)][item] for k in return_keys}
        else:
            return {k: self.dataset[k][item] for k in return_keys}

    def __len__(self):

        return self.dataset['question1_input_ids'].shape[0]


def load_split_qqp_dataset_and_tokenizer_from_disk(data_path="data",):

    with open(os.path.join(data_path, "qqp_train"), "rb") as f:
        qqp_train = pickle.load(f)

    with open(os.path.join(data_path, "qqp_dev"), "rb") as f:
        qqp_eval = pickle.load(f)
    
    with open(os.path.join(data_path, "qqp_paraphrase_vocab_dict"), "rb") as f:
        vocab_dict = pickle.load(f)

    return qqp_train, qqp_eval, vocab_dict
