# https://github.com/nusnlp/PtrNetDecoding4JERE
import sys
import os
import numpy as np
import random

import pickle
import datetime
from torch.nn import Parameter
from collections import OrderedDict
from tqdm import tqdm
from recordclass import recordclass
from typing import Dict, List, Tuple, Set, Optional
from functools import partial

from openjere.metrics import F1_triplet
from openjere.models.abc_model import ABCModel
from openjere.config.const import *

import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json


# enc_type = ['LSTM', 'GCN', 'LSTM-GCN'][0]
# att_type = ['None', 'Unigram', 'N-Gram-Enc'][1]
def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


random_seed = 44
n_gpu = torch.cuda.device_count()
set_random_seeds(random_seed)


batch_size = 64
num_epoch = 30
max_src_len = 100
max_trg_len = 50
# embedding_file = os.path.join(src_data_folder, 'w2v.txt')
update_freq = 1

copy_on = True

gcn_num_layers = 3
word_embed_dim = 300
word_min_freq = 2
char_embed_dim = 50
char_feature_size = 50
conv_filter_size = 3
max_word_len = 10

# enc_inp_size = word_embed_dim + char_feature_size
enc_inp_size = word_embed_dim
enc_hidden_size = 256 #word_embed_dim
dec_inp_size = word_embed_dim #enc_hidden_size
dec_hidden_size = 256#dec_inp_size

drop_rate = 0.5
layers = 1
early_stop_cnt = 5
sample_cnt = 0
Sample = recordclass("Sample", "Id SrcLen SrcWords TrgLen TrgWords AdjMat")



def get_pred_words(preds, attns, src_words, word_vocab, rev_word_vocab):
    pred_words = []
    for i in range(0, max_trg_len):
        word_idx = preds[i]
        if word_vocab[EOS] == word_idx:
            pred_words.append(EOS)
            break
        elif copy_on and word_vocab[OOV] == word_idx:
            word_idx = attns[i]
            pred_words.append(src_words[word_idx])
        else:
            pred_words.append(rev_word_vocab[word_idx])
    return pred_words


def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums += 1
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj = (_adj > t).astype(int)
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    # adj = A.clone()
    return adj

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, is_bidirectional, drop_out_rate):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.is_bidirectional = is_bidirectional
        self.drop_rate = drop_out_rate
        # self.char_embeddings = CharEmbeddings(len(char_vocab), char_embed_dim, drop_rate)
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            self.layers,
            batch_first=True,
            bidirectional=self.is_bidirectional,
        )
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, words_input, tokens_len, char_seq, adj, is_training=False):
        words_input = words_input
        embedded = nn.utils.rnn.pack_padded_sequence(words_input, tokens_len, batch_first=True)
        outputs, hc = self.lstm(embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        outputs = torch.add(*outputs.chunk(2, dim=2))/2
        outputs = self.dropout(outputs)
        hc = (self.dropout(hc[0]), self.dropout(hc[1]))
        return outputs, hc


class Attention(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.linear_ctx = nn.Linear(self.input_dim, self.out_dim, bias=False)
        self.linear_query = nn.Linear(self.input_dim, self.out_dim, bias=True)
        self.v = nn.Linear(self.out_dim, 1)
    def forward(self, s_prev, enc_hs, src_mask=None):
        uh = self.linear_ctx(enc_hs)
        wq = self.linear_query(s_prev)
        wquh = torch.tanh(wq + uh)
        attn_weights = self.v(wquh).squeeze()
        if src_mask is not None:
            attn_weights.data.masked_fill_(src_mask.data, -float("inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        ctx = torch.bmm(attn_weights.unsqueeze(1), enc_hs).squeeze()
        return ctx, attn_weights

class Decoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, layers, drop_out_rate, max_length, vocab_length, is_backward=False
    ):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.drop_rate = drop_out_rate
        self.max_length = max_length

        self.attention = Attention(hidden_dim, hidden_dim)
        self.is_backward = is_backward
        if not self.is_backward:
            self.bidirection_attention = Attention(hidden_dim, hidden_dim)
        if is_backward:
            self.lstm = nn.LSTMCell(self.input_dim+enc_hidden_size, self.hidden_dim, self.layers)
            self.ent_out = nn.Linear(hidden_dim, vocab_length)
        else:
            self.lstm = nn.LSTMCell(self.input_dim+2*enc_hidden_size, self.hidden_dim, self.layers)
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(
        self, y_prev, h_prev, enc_hs, src_word_embeds, src_mask, h_backward=None, is_training=False
    ):
        src_time_steps = enc_hs.size()[1]

        s_prev = h_prev[0]
        s_prev = s_prev.unsqueeze(1)
        s_prev = s_prev.repeat(1, src_time_steps, 1)
        ctx, attn_weights = self.attention(s_prev, enc_hs, src_mask)

        y_prev = y_prev.squeeze()
        if not self.is_backward:
            back_time_steps = h_backward.size()[1]
            s_prev = h_prev[0]
            s_prev = s_prev.unsqueeze(1)
            s_prev = s_prev.repeat(1, back_time_steps, 1)
            ctx_backward, _ = self.bidirection_attention(s_prev, h_backward)
            s_cur = torch.cat((y_prev, ctx_backward, ctx), 1)
        else:
            s_cur = torch.cat((y_prev, ctx), 1)
        hidden, cell_state = self.lstm(s_cur, h_prev)
        hidden = self.dropout(hidden)
        if self.is_backward:
            output = self.ent_out(hidden)
        else:
            output = hidden
        return output, (hidden, cell_state), attn_weights


class WDec(ABCModel):
    def __init__(self, hyper):
        super(WDec, self).__init__()
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, "word_vocab.json"), "r", encoding="utf-8")
        )
        self.relation_vocab = json.load(
            open(
                os.path.join(self.data_root, "relation_vocab.json"),
                "r",
                encoding="utf-8",
            )
        )
        self.rev_vocab = {v: k for k, v in self.word_vocab.items()}

        num_classes = len(self.relation_vocab)
        _adj = gen_A(num_classes, 0, os.path.join(self.data_root, 'adj.pkl'))
        self.A = Parameter(torch.from_numpy(_adj).float()).cuda(self.gpu)
        # self.gc1 = GraphConvolution(self.hyper.emb_size, self.hyper.emb_size)
        # self.gc2 = GraphConvolution(self.hyper.emb_size, self.hyper.emb_size)
        self.gc1 = GraphConvolution(dec_hidden_size, dec_hidden_size)
        self.gc2 = GraphConvolution(dec_hidden_size, dec_hidden_size)      
        self.relu = nn.LeakyReLU(0.2)        
        self.word_embeddings = nn.Embedding(len(self.word_vocab), self.hyper.emb_size)
        use_pretrain_embedding = True
        if use_pretrain_embedding:
            from openjere.models.pretrain import PretranEmbedding, load_pretrained_embedding
            self.pe = PretranEmbedding(self.hyper)
            self.word_embeddings.weight.data.copy_(
            load_pretrained_embedding(self.word_vocab, self.pe))
        self.word_cls = nn.Embedding(len(self.word_vocab)- len(self.relation_vocab), dec_hidden_size)
        self.rel_cls = nn.Embedding(len(self.relation_vocab), dec_hidden_size)
       
        # self.word_embeddings = WordEmbeddings(len(word_vocab), word_embed_dim, word_embed_matrix, drop_rate)
        # self.encoder = Encoder(
        #     enc_inp_size, int(enc_hidden_size / 2), layers, True, drop_rate
        # )
        self.encoder = Encoder(
            enc_inp_size, int(enc_hidden_size), layers, True, drop_rate
        )
        self.backward_decoder = Decoder(
            dec_inp_size,
            dec_hidden_size,
            layers,
            drop_rate,
            max_trg_len,
            len(self.word_vocab),
            is_backward=True
        )
        self.forward_decoder = Decoder(
            dec_inp_size,
            dec_hidden_size,
            layers,
            drop_rate,
            max_trg_len,
            len(self.word_vocab),
        )
        self.criterion = nn.NLLLoss(ignore_index=0)
        # self.criterion = nn.NLLLoss()
        self.metrics = F1_triplet()
        self.get_metric = self.metrics.get_metric

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:

        src_words_seq = sample.tokens_id.cuda(self.gpu)
        # src_chars_seq = sample.src_chars_seq
        src_mask = sample.src_words_mask.cuda(self.gpu)
        trg_vocab_mask = sample.trg_vocab_mask.cuda(self.gpu)
        SrcWords = list(map(lambda x: self.hyper.tokenizer(x), sample.text))
        B = len(SrcWords)
        if True:
            trg_words_seq = sample.trg_words.cuda(self.gpu)
            back_trg_words_seq = sample.back_trg_words.cuda(self.gpu)
            target = sample.target.cuda(self.gpu)
            back_target = sample.back_target.cuda(self.gpu)
            trg_word_embeds = self.word_embeddings(trg_words_seq)
            back_trg_word_embeds = self.word_embeddings(back_trg_words_seq)

        sos = torch.LongTensor(B * [self.word_vocab[SOS]]).cuda(self.gpu)
        sos = self.word_embeddings(sos)

        # eos = torch.LongTensor(B * [self.word_vocab[EOS]]).cuda(self.gpu)
        # eos = self.word_embeddings(eos)

        src_word_embeds = self.word_embeddings(src_words_seq)

        batch_len = src_word_embeds.size()[0]
        if is_train:
            time_steps = trg_word_embeds.size()[1] - 1
        else:
            time_steps = max_trg_len

        encoder_output, hc = self.encoder(src_word_embeds, list(sample.length), None, None, is_train) 
        h0_backward = hc[0][1,:,:]
        c0_backward = hc[1][1,:,:]
        dec_hid_backward = (h0_backward, c0_backward)

        
        using_gcn = True
        if using_gcn:
            adj = gen_adj(self.A).detach()
            rel_cls_weight = self.gc1(self.rel_cls.weight, adj)
            rel_cls_weight = self.relu(rel_cls_weight)
            rel_cls_weight = self.gc2(rel_cls_weight, adj)
            cls_weight = torch.cat((self.word_cls.weight, rel_cls_weight), dim=0)
        else:

            cls_weight = torch.cat((self.word_cls.weight, self.rel_cls.weight), dim=0)

        cls_weight = cls_weight.t()
        output = {"text": sample.text}
        h_backward = h0_backward.unsqueeze(1)
        if is_train:
            dec_inp_backward = back_trg_word_embeds[:, 0, :]
            dec_out_backward, dec_hid_backward, dec_attn_backward = self.backward_decoder(
                dec_inp_backward, dec_hid_backward, encoder_output, src_word_embeds, src_mask, is_train
            )
            h_backward = torch.cat((dec_hid_backward[0].unsqueeze(1), h_backward), 1)
            # dec_out_backward = torch.mm(dec_out_backward, cls_weight)
            dec_out_backward = dec_out_backward.view(-1, len(self.word_vocab))
            dec_out_backward = F.log_softmax(dec_out_backward, dim=-1)
            dec_out_v_backward, dec_out_i_backward = dec_out_backward.topk(1)
            dec_out_backward = dec_out_backward.unsqueeze(1)
            
            for t in range(1, time_steps):
                dec_inp_backward = back_trg_word_embeds[:, t, :]
                cur_dec_out_backward, dec_hid_backward, dec_attn_backward = self.backward_decoder(
                    dec_inp_backward,
                    dec_hid_backward,
                    encoder_output,
                    src_word_embeds,
                    src_mask,
                    is_train,
                )
                h_backward = torch.cat((dec_hid_backward[0].unsqueeze(1), h_backward), 1)
                # cur_dec_out_backward = torch.mm(cur_dec_out_backward, cls_weight)
                cur_dec_out_backward = cur_dec_out_backward.view(-1, len(self.word_vocab))
                dec_out_backward = torch.cat(
                    (dec_out_backward, F.log_softmax(cur_dec_out_backward, dim=-1).unsqueeze(1)), 1
                )
                cur_dec_out_v_backward, cur_dec_out_i_backward = cur_dec_out_backward.topk(1)
                dec_out_i_backward = torch.cat((dec_out_i_backward, cur_dec_out_i_backward), 1)
        else:
            dec_inp_backward = sos
            dec_out_backward, dec_hid_backward, dec_attn_backward = self.backward_decoder(
                dec_inp_backward, dec_hid_backward, encoder_output, src_word_embeds, src_mask, is_train
            )
            h_backward = torch.cat((dec_hid_backward[0].unsqueeze(1), h_backward), 1)
            # dec_out_backward = torch.mm(dec_out_backward, cls_weight)
            dec_out_backward = dec_out_backward.view(-1, len(self.word_vocab))
            if copy_on:
                dec_out_backward.data.masked_fill_(trg_vocab_mask.data, -float("inf"))
            dec_out_backward = F.log_softmax(dec_out_backward, dim=-1)
            topv_backward, topi_backward = dec_out_backward.topk(1)
            dec_out_v_backward, dec_out_i_backward = dec_out_backward.topk(1)
            dec_attn_v_backward, dec_attn_i_backward = dec_attn_backward.topk(1)

            for t in range(1, time_steps):
                dec_inp_backward = self.word_embeddings(topi_backward.squeeze().detach())
                cur_dec_out_backward, dec_hid_backward, cur_dec_attn_backward = self.backward_decoder(
                    dec_inp_backward,
                    dec_hid_backward,
                    encoder_output,
                    src_word_embeds,
                    src_mask,
                    is_train,
                )
                h_backward = torch.cat((dec_hid_backward[0].unsqueeze(1), h_backward), 1)
                # cur_dec_out_backward = torch.mm(cur_dec_out_backward, cls_weight)
                cur_dec_out_backward = cur_dec_out_backward.view(-1, len(self.word_vocab))
                if copy_on:
                    cur_dec_out_backward.data.masked_fill_(trg_vocab_mask.data, -float("inf"))
                cur_dec_out_backward = F.log_softmax(cur_dec_out_backward, dim=-1)
                topv_backward, topi_backward = cur_dec_out_backward.topk(1)
                cur_dec_out_v_backward, cur_dec_out_i_backward = cur_dec_out_backward.topk(1)
                dec_out_i_backward = torch.cat((dec_out_i_backward, cur_dec_out_i_backward), 1)
                cur_dec_attn_v_backward, cur_dec_attn_i_backward = cur_dec_attn_backward.topk(1)
                dec_attn_i_backward = torch.cat((dec_attn_i_backward, cur_dec_attn_i_backward), 1)

        if is_train:

            dec_out_backward = dec_out_backward.view(-1, len(self.word_vocab))
            tst = (dec_out_i_backward == back_target).sum(1)
            back_target = back_target.view(-1, 1).squeeze()
            backward_loss = self.criterion(dec_out_backward, back_target)

            # output["loss"] = loss

            # output["description"] = partial(self.description, output=output)

        else:
            spo_gold = sample.spo_gold
            output["spo_gold"] = spo_gold
            preds_backward = list(dec_out_i_backward.data.cpu().numpy())
            target_backward = list(back_target.data.cpu().numpy())
            attns_backward = list(dec_attn_i_backward.data.cpu().numpy())
            result_backward = []
            seq_backward = []
            for i in range(0, len(preds_backward)):
                pred_words_backward = get_pred_words(
                    preds_backward[i], attns_backward[i], SrcWords[i], self.word_vocab, self.rev_vocab
                )
                seq_backward.append(pred_words_backward)
                decoded_triplets_backward = self.seq2triplet_back(pred_words_backward)
                result_backward.append(decoded_triplets_backward)
            output["decode_result"] = result_backward
            output["back_seq"] = seq_backward

        h0 = hc[0][0,:,:]
        c0 = hc[1][0,:,:]
        dec_hid = (h0, c0)
        
        if is_train:
            dec_inp = trg_word_embeds[:, 0, :]
            dec_out, dec_hid, dec_attn = self.forward_decoder(
                dec_inp, dec_hid, encoder_output, src_word_embeds, src_mask, h_backward=h_backward, is_training=is_train
            )
            dec_out = torch.mm(dec_out, cls_weight)
            dec_out = dec_out.view(-1, len(self.word_vocab))
            dec_out = F.log_softmax(dec_out, dim=-1)
            dec_out_v, dec_out_i = dec_out.topk(1)
            dec_out = dec_out.unsqueeze(1)
            for t in range(1, time_steps):
                dec_inp = trg_word_embeds[:, t, :]
                cur_dec_out, dec_hid, cur_dec_attn = self.forward_decoder(
                    dec_inp,
                    dec_hid,
                    encoder_output,
                    src_word_embeds,
                    src_mask,
                    h_backward=h_backward,
                    is_training=is_train
                )
                cur_dec_out = torch.mm(cur_dec_out, cls_weight)
                cur_dec_out = cur_dec_out.view(-1, len(self.word_vocab))
                dec_out = torch.cat(
                    (dec_out, F.log_softmax(cur_dec_out, dim=-1).unsqueeze(1)), 1
                )
                cur_dec_out_v, cur_dec_out_i = cur_dec_out.topk(1)
                dec_out_i = torch.cat((dec_out_i, cur_dec_out_i), 1)
        else:
            dec_inp = sos
            dec_out, dec_hid, dec_attn = self.forward_decoder(
                dec_inp, dec_hid, encoder_output, src_word_embeds, src_mask, h_backward=h_backward, is_training=is_train
            )
            dec_out = torch.mm(dec_out, cls_weight)
            dec_out = dec_out.view(-1, len(self.word_vocab))
            if copy_on:
                dec_out.data.masked_fill_(trg_vocab_mask.data, -float("inf"))
            dec_out = F.log_softmax(dec_out, dim=-1)
            topv, topi = dec_out.topk(1)
            dec_out_v, dec_out_i = dec_out.topk(1)
            dec_attn_v, dec_attn_i = dec_attn.topk(1)

            for t in range(1, time_steps):
                dec_inp = self.word_embeddings(topi.squeeze().detach())
                cur_dec_out, dec_hid, cur_dec_attn = self.forward_decoder(
                    dec_inp,
                    dec_hid,
                    encoder_output,
                    src_word_embeds,
                    src_mask,
                    h_backward=h_backward,
                    is_training=is_train
                )
                cur_dec_out = torch.mm(cur_dec_out, cls_weight)
                cur_dec_out = cur_dec_out.view(-1, len(self.word_vocab))
                if copy_on:
                    cur_dec_out.data.masked_fill_(trg_vocab_mask.data, -float("inf"))
                cur_dec_out = F.log_softmax(cur_dec_out, dim=-1)
                topv, topi = cur_dec_out.topk(1)
                cur_dec_out_v, cur_dec_out_i = cur_dec_out.topk(1)
                dec_out_i = torch.cat((dec_out_i, cur_dec_out_i), 1)
                cur_dec_attn_v, cur_dec_attn_i = cur_dec_attn.topk(1)
                dec_attn_i = torch.cat((dec_attn_i, cur_dec_attn_i), 1)

        if is_train:

            dec_out = dec_out.view(-1, len(self.word_vocab))
            tst = (dec_out_i == target).sum(1)
            target = target.view(-1, 1).squeeze()
            forward_loss = self.criterion(dec_out, target)

            output["loss"] = 0.3 * backward_loss +  0.7 * forward_loss

            output["description"] = partial(self.description, output=output)

        else:
            spo_gold = sample.spo_gold
            output["spo_gold"] = spo_gold
            preds = list(dec_out_i.data.cpu().numpy())
            attns = list(dec_attn_i.data.cpu().numpy())
            result = []
            seq = []
            for i in range(0, len(preds)):
                pred_words = get_pred_words(
                    preds[i], attns[i], SrcWords[i], self.word_vocab, self.rev_vocab
                )
                seq.append(pred_words)
                decoded_triplets = self.seq2triplet(pred_words)
                result.append(decoded_triplets)
            output["decode_result"] += result
            output["seq"] = seq
        return output

    def seq2triplet(self, pred_words: List[str]) -> List[Dict[str, str]]:
        result = []
        pred_words = self.hyper.join(pred_words[:-1])
        for t in pred_words.split(SEP_VERTICAL_BAR):
            parts = t.split(SEP_SEMICOLON)
            if len(parts) != 3:
                continue
            em1 = self.hyper.join(parts[0].strip().split(" "))
            em2 = self.hyper.join(parts[1].strip().split(" "))
            rel = parts[2].strip()

            if len(em1) == 0 or len(em2) == 0 or len(rel) == 0:
                continue

            if rel not in self.relation_vocab.keys():
                continue

            triplet = {"subject": em1, "predicate": rel, "object": em2}
            result.append(triplet)
        return result

    def seq2triplet_back(self, pred_words: List[str]) -> List[Dict[str, str]]:
        result = []
        pred_words = self.hyper.join(pred_words[:-1])
        for t in pred_words.split(SEP_VERTICAL_BAR):
            parts = t.split(SEP_SEMICOLON)
            if len(parts) != 3:
                continue
            em1 = self.hyper.join(parts[2].strip().split(" ")[::-1])
            em2 = self.hyper.join(parts[1].strip().split(" ")[::-1])
            rel = parts[0].strip()

            if len(em1) == 0 or len(em2) == 0 or len(rel) == 0:
                continue

            if rel not in self.relation_vocab.keys():
                continue

            triplet = {"subject": em1, "predicate": rel, "object": em2}
            result.append(triplet)
        return result

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, epoch: {}/{}:".format(
            output["loss"].item(), epoch, epoch_num,
        )

    def run_metrics(self, output):
        self.metrics(output["decode_result"], output["spo_gold"], output["text"])
