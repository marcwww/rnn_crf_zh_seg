import torch
from torch import nn
from macros import *
import utils

torch.backends.cudnn.benchmark=True

class BiLSTM_CRF(nn.Module):

    def __init__(self, voc_size,
                 idx2tag, tag2idx,
                 emb_dim, hdim):
        super(BiLSTM_CRF, self).__init__()
        self.emb_dim = emb_dim
        self.hdim = hdim
        self.voc_size = voc_size
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag
        self.tag_size = len(tag2idx)

        self.word_embeds = nn.Embedding(voc_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hdim // 2,
                            num_layers=1, bidirectional=True)

        self.hid2tag = nn.Linear(hdim, self.tag_size)

        # (i,j) element: P(j->i)
        self.trans = nn.Parameter(torch.randn(self.tag_size, self.tag_size))

        self.trans.data[tag2idx[TAG_BOS], :] = -10000
        self.trans.data[:, tag2idx[TAG_EOS]] = -10000
        self.trans.data[tag2idx[PAD_TAG], :] = 0

    def init_hidden(self, bsz):
        return (torch.zeros(2, bsz, self.hdim // 2),
                torch.zeros(2, bsz, self.hdim // 2))

    def _forward_alg(self, fts):
        bsz, tag_size = fts.shape[1], fts.shape[2]
        # init_alphas: (bsz, tag_size)
        init_alphas = torch.full((bsz,self.tag_size), -10000.)
        init_alphas[:][self.tag2idx[TAG_BOS]] = 0.

        # forward_var: (bsz, tag_size)
        forward_var = init_alphas

        # fts: (seq_len, bsz, tag_size)
        # ft: (bsz, tag_size)
        # trans: (tag_size, tag_size)
        for ft in fts:
            alphas_t = []
            for next_tag in range(self.tag_size):
                # emit_score: (bsz, 1)
                emit_score = ft[:, next_tag].unsqueeze(-1)
                # trans_score: (bsz, tag_size)
                trans_score = self.trans[next_tag].expand(bsz, tag_size)
                # next_tag_var: (bsz, tag_size)
                next_tag_var = forward_var + trans_score + emit_score

                alphas_t.append(utils.log_sum_exp(next_tag_var))
            # alphas_t(list): tag_size * (bsz, 1)
            # forward_var: (bsz, tag_size)
            forward_var = torch.cat(alphas_t, dim=1)

        terminal_var = forward_var + self.trans[self.tag2idx[TAG_EOS]]
        alpha = utils.log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_fts(self, sen):
        bsz = sen.shape[1]
        self.hid = self.init_hidden(bsz)
        embeds = self.word_embeds(sen)
        lstm_out, self.hid = self.lstm(embeds, self.hid)
        lstm_fts = self.hid2tag(lstm_out)
        return lstm_fts

    def _score_sen(self, fts, tags):
        bsz, tag_size = fts.shape[1], fts.shape[2]
        score = torch.zeros(bsz)
        # tags = torch.cat([torch.
        #                  LongTensor([self.tag2idx[TAG_BOS]]),
        #                   tags])

        # fts: (seq_len, bsz, tag_size)
        # ft: (bsz, tag_size)
        # trans: (tag_size, tag_size)
        # score: (bsz)
        # trans: (bsz * tag_size * tag_size)
        trans = self.trans.expand(bsz, tag_size, tag_size).\
            contiguous().view(-1)
        mtrx_elems = tag_size*tag_size

        # tags: (seq_len, bsz)
        for i, ft in enumerate(fts):
            indices_trans = torch.LongTensor(range(0, bsz * mtrx_elems, mtrx_elems)) + \
                      tags[i+1]*tag_size+ \
                      tags[i]
            indices_ft = torch.LongTensor(range(0, bsz * tag_size, tag_size)) + \
                         tags[i+1]

            score = score + \
                    trans[indices_trans]+ \
                    ft.view(-1)[indices_ft]

        indices_trans = torch.LongTensor(range(0, bsz * mtrx_elems, mtrx_elems)) + \
                  self.tag2idx[TAG_EOS] * tag_size + \
                  tags[-1]

        score = score + trans[indices_trans]
        return score

    def _viterbi_decode(self, fts):
        back_ptrs = []

        bsz, tag_size = fts.shape[1], fts.shape[2]

        # Initialize the viterbi variables
        init_vvars = torch.full((bsz, self.tag_size), -10000.)
        init_vvars[:, self.tag2idx[TAG_BOS]] = 0.

        # forward_var: (bsz, tag_size)
        # forward_var at step i holds the viterbi variables at step i-1
        forward_var = init_vvars

        # ft: (bsz, tag_size)
        for ft in fts:
            bptrs_t = []
            viterbi_vars_t = []

            for next_tag in range(self.tag_size):
                # next_tag_var: (bsz, tag_size)
                next_tag_var = forward_var + self.trans[next_tag]

                # best_tag_id: (bsz)
                best_tag_id = utils.argmax(next_tag_var)
                bptrs_t.append(best_tag_id.unsqueeze(-1))

                indices = torch.LongTensor(range(0, bsz * tag_size, tag_size)) + \
                          best_tag_id
                viterbi_vars_t.append(next_tag_var.view(-1)[indices].unsqueeze(-1))

            # viterbi_vars_t(list): tag_size * (bsz, 1)
            forward_var = (torch.cat(viterbi_vars_t, dim=1) + ft)
            back_ptrs.append(bptrs_t)

        terminal_var = forward_var + self.trans[self.tag2idx[TAG_EOS]]
        # best_tag_id: (bsz)
        best_tag_id = utils.argmax(terminal_var)
        indices = torch.LongTensor(range(0, bsz * tag_size, tag_size)) + \
                  best_tag_id
        # path_score: (bsz)
        path_score = terminal_var.view(-1)[indices]

        best_path = [best_tag_id.unsqueeze(-1)]
        # back_ptrs(list): seq_len * (tag_size * (bsz))
        for bptrs_t in reversed(back_ptrs):
            # bptrs_t(list): tag_size * (bsz)
            bptrs_t = torch.cat(bptrs_t, dim=1)
            indices = torch.LongTensor(range(0, bsz * tag_size, tag_size)) + \
                      best_tag_id
            # best_tag_id: (bsz)
            best_tag_id = bptrs_t.view(-1)[indices]
            best_path.append(best_tag_id.unsqueeze(-1))

        # start: (bsz)
        start = best_path.pop()
        assert torch.sum(start - torch.LongTensor([self.tag2idx[TAG_BOS]] * bsz)).\
                   item() == 0
        best_path.reverse()
        return path_score, best_path

    def neg_logllihood(self, sen, tags):
        fts = self._get_lstm_fts(sen)
        bsz = fts.shape[1]
        forward_score = self._forward_alg(fts)
        gold_score = self._score_sen(fts, tags)
        return (torch.sum(forward_score - gold_score))/bsz

    def forward(self, sen):
        lstm_fts = self._get_lstm_fts(sen)
        score, tag_seq = self._viterbi_decode(lstm_fts)
        return score, tag_seq