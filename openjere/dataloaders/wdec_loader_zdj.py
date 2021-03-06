from recordclass import recordclass
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from .abc_dataset import Abstract_dataset
from openjere.config.const import (
    PAD,
    seq_padding,
    OOV,
    SOS,
    EOS,
    SEP_SEMICOLON,
    SEP_VERTICAL_BAR,
)


import numpy as np
import os
import json
import random

import torch

Sample = recordclass("Sample", "Id SrcLen SrcWords TrgLen TrgWords AdjMat")


def get_target_vocab_mask(src_words, word_vocab, relations):
    mask = []
    for i in range(0, len(word_vocab)):
        mask.append(1)
    for word in src_words:
        if word in word_vocab:
            mask[word_vocab[word]] = 0
    for rel in relations:
        mask[word_vocab[rel]] = 0

    mask[word_vocab[OOV]] = 0
    mask[word_vocab[EOS]] = 0
    # mask[word_vocab[SOS]] = 0
    mask[word_vocab[SEP_SEMICOLON]] = 0
    mask[word_vocab[SEP_VERTICAL_BAR]] = 0
    return mask


class WDec_Dataset(Abstract_dataset):
    def __init__(self, hyper, dataset):

        super(WDec_Dataset, self).__init__(hyper, dataset)

        self.seq_list = []
        self.text_list = []
        self.spo_list = []

        for line in open(os.path.join(self.data_root, dataset), "r"):
            line = line.strip("\n")
            instance = json.loads(line)

            self.seq_list.append(instance["seq"])
            self.text_list.append(instance["text"])
            self.spo_list.append(instance["spo_list"])

    def __getitem__(self, index):

        seq = self.seq_list[index]
        text = self.text_list[index]
        spo = self.spo_list[index]

        tokens_id = self.text2id(text)
        trg_words, back_trg_words, trg_line, back_trg_line = self.seq2id(seq)    

        trg_vocab_mask = get_target_vocab_mask(
            self.hyper.tokenizer(text), self.word_vocab, self.relation_vocab.keys()
        )

        return (
            tokens_id,
            trg_words,
            len(tokens_id),
            len(trg_words),
            trg_vocab_mask,
            spo,
            text,
            trg_line,
            back_trg_line,
            back_trg_words
        )

    def __len__(self):
        return len(self.text_list)

    def text2id(self, text: List[str]) -> torch.tensor:
        oov = self.word_vocab[OOV]
        text_list = list(
            map(lambda x: self.word_vocab.get(x, oov), self.tokenizer(text))
        )
        return text_list

    def seq2id(self, seq):
        oov = self.word_vocab[OOV]
        tuples = seq.strip().split(SEP_VERTICAL_BAR)
        if True:
            new_tuples = []
            for t in tuples:
                tmp = t.split(SEP_SEMICOLON)
                t_c = (" " + SEP_SEMICOLON + " ").join((tmp[0], tmp[2], tmp[1]))
                new_tuples.append(t_c)
            tuples = new_tuples
        tuples_back = tuples[::-1]
        # random.shuffle(tuples)
        def get_trg(tuples):
            # tuples = [(" " + SEP_SEMICOLON + " ").join(tuple.split(SEP_SEMICOLON)[::-1]) for tuple in tuples]            
            new_trg_line = (" " + SEP_VERTICAL_BAR + " ").join(tuples)
            assert len(seq.split()) == len(new_trg_line.split())
            trg_line = new_trg_line

            trg_words = trg_line.split()
            trg_words = [SOS] + trg_words + [EOS]
            trg_words = list(map(lambda x: self.word_vocab.get(x, oov), trg_words))
            return trg_words, trg_line
        def get_trg_back(tuples):

            # tuples = [(" " + SEP_SEMICOLON + " ").join(tuple.split(SEP_SEMICOLON)[::-1]) for tuple in tuples] 
            new_trg_line = (" " + SEP_VERTICAL_BAR + " ").join(tuples)
            assert len(seq.split()) == len(new_trg_line.split())
            trg_line = new_trg_line

            trg_words = trg_line.split()#[::-1]
            trg_words = [SOS] + trg_words + [EOS]
            trg_words = list(map(lambda x: self.word_vocab.get(x, oov), trg_words))
            return trg_words, trg_line
        trg_words, trg_line = get_trg(tuples)
        back_trg_words, back_trg_line = get_trg_back(tuples_back)
        return trg_words, back_trg_words, trg_line, back_trg_line

class Batch_reader(object):
    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        transposed_data = list(zip(*data))
        self.tokens_id = torch.tensor(seq_padding(transposed_data[0]))#?????????padding??????id

        # TODO
        self.trg_words = torch.tensor(seq_padding(transposed_data[1]))#?????????padding?????????id
        # back_tri = [[item[0]] + item[1:-1][::-1] + [item[-1]] for item in transposed_data[1]]
        back_tri = transposed_data[-1]
        self.back_trg_words = torch.tensor(seq_padding(back_tri))#?????????padding?????????id
        self.target = torch.tensor(seq_padding([ins[1:] for ins in transposed_data[1]]))#trg_words?????????[SOS]???????????????
        self.back_target = torch.tensor(seq_padding([ins[1:] for ins in back_tri]))
        self.src_words_mask = torch.eq(self.tokens_id, 0)#padding???mask
        # self.seq_words_mask = torch.gt(self.seq_id, 0)

        self.en_len = transposed_data[2]#????????????????????????   ???padding
        self.de_len = transposed_data[3]#???????????????????????????   ???padding
        self.trg_vocab_mask = torch.tensor(transposed_data[4]).bool()#???????????????????????????????????????mask
        self.spo_gold = transposed_data[5]#spo
        self.text = transposed_data[6]#??????
        self.length = self.en_len
        self.seq = transposed_data[7]
        self.back_seq = transposed_data[8]

    def pin_memory(self):
        self.tokens_id = self.tokens_id.pin_memory()
        self.trg_words = self.trg_words.pin_memory()
        self.target = self.target.pin_memory()
        self.src_words_mask = self.src_words_mask.pin_memory()
        # self.seq_words_mask = self.seq_words_mask.pin_memory()
        self.trg_vocab_mask = self.trg_vocab_mask.pin_memory()
        return self


def collate_fn(batch):
    return Batch_reader(batch)


WDec_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)
