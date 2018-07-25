import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_idx):
    idices = [to_idx[w] for w in seq]
    return torch.LongTensor(idices)
    # return torch.LongTensor(idices)

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # s_max + log(sum(exp(vec-s_max)))
    # = s_max + log(sum(exp(vec)/exp(s_max)))
    # = s_max + log(1/exp(s_max)\cdot sum(exp(vec)))
    # = s_max + -s_max + log(sum(exp(vec)))
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, voc_size, tag2idx, emb_dim, hdim):
        super(BiLSTM_CRF, self).__init__()
        self.emb_dim = emb_dim
        self.hdim = hdim
        self.voc_size = voc_size
        self.tag2idx = tag2idx
        self.tag_size = len(tag2idx)

        self.word_embeds = nn.Embedding(voc_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hdim // 2,
                            num_layers=1, bidirectional=True)

        self.hid2tag = nn.Linear(hdim, self.tag_size)

        self.trans = nn.Parameter(torch.randn(self.tag_size, self.tag_size))

        self.trans.data[tag2idx[START_TAG], :] = -10000
        self.trans.data[:, tag2idx[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hdim // 2),
                torch.randn(2, 1, self.hdim // 2))

    def _forward_alg(self, fts):
        init_alphas = torch.full((1,self.tag_size), -10000.)
        init_alphas[0][self.tag2idx[START_TAG]] = 0.

        forward_var = init_alphas

        for ft in fts:
            alphas_t = []
            for next_tag in range(self.tag_size):
                emit_score = ft[next_tag].view(1, -1).\
                    expand(1, self.tag_size)

                trans_score = self.trans[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score

                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.trans[self.tag2idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_fts(self, sen):
        self.hid = self.init_hidden()
        embeds = self.word_embeds(sen).view(len(sen), 1, -1)
        lstm_out, self.hid = self.lstm(embeds, self.hid)
        lstm_out = lstm_out.view(len(sen), self.hdim)
        lstm_fts = self.hid2tag(lstm_out)
        return lstm_fts

    def _score_sen(self, fts, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.
                         LongTensor([self.tag2idx[START_TAG]]),
                          tags])
        for i, ft in enumerate(fts):
            score = score + \
                    self.trans[tags[i + 1], tags[i]] + \
                    ft[tags[i + 1]]

        score = score + self.trans[self.tag2idx[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, fts):
        back_ptrs = []

        # Initialize the viterbi variables
        init_vvars = torch.full((1, self.tag_size), -10000.)
        init_vvars[0][self.tag2idx[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables at step i-1
        forward_var = init_vvars
        for ft in fts:
            bptrs_t = []
            viterbi_vars_t = []

            for next_tag in range(self.tag_size):
                next_tag_var = forward_var + self.trans[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbi_vars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbi_vars_t) + ft).view(1, -1)
            back_ptrs.append(bptrs_t)

        terminal_var = forward_var + self.trans[self.tag2idx[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(back_ptrs):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag2idx[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_logllihood(self, sen, tags):
        fts = self._get_lstm_fts(sen)
        forward_score = self._forward_alg(fts)
        gold_score = self._score_sen(fts, tags)
        return forward_score - gold_score

    def forward(self, sen):
        lstm_fts = self._get_lstm_fts(sen)
        score, tag_seq = self._viterbi_decode(lstm_fts)
        return score, tag_seq

training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

w2idx = {}
for sen, _ in training_data:
    for w in sen:
        if w not in w2idx:
            w2idx[w] = len(w2idx)

tag2idx = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
model = BiLSTM_CRF(len(w2idx), tag2idx, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)

with torch.no_grad():
    precheck_sen = prepare_sequence(training_data[0][0], w2idx)
    precheck_tags = torch.LongTensor([tag2idx[t] for t in training_data[0][1]])
    print(model(precheck_sen))

for epoch in range(300):
    for sen, tags in training_data:
        model.zero_grad()
        sen_in = prepare_sequence(sen, w2idx)
        tars = torch.LongTensor([tag2idx[t] for t in tags])
        loss = model.neg_logllihood(sen_in, tars)
        print(loss.item())
        loss.backward()
        optimizer.step()

with torch.no_grad():
    precheck_sen = prepare_sequence(training_data[0][0], w2idx)
    print(model(precheck_sen))






