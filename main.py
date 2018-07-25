import preproc
import argparse
import opts
import nets
from macros import *
from torch import optim
import torch
import os
from sklearn.metrics import precision_score, \
    recall_score, \
    f1_score

def insert(dictionary, tag):
    dictionary[tag] = len(dictionary)

def progress_bar(percent, last_loss, epoch):
    """Prints the progress until the next report."""
    fill = int(percent * 40)
    print("\r[{}{}]: {:.4f}/epoch {:d} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), percent, epoch, last_loss), end='')

def train(model, iters, opt, optim):
    train_iter = iters['train']
    valid_iter = iters['valid']

    # valid(model, valid_iter, opt)

    for epoch in range(opt.nepoch):
        for i, sample in enumerate(train_iter):
            model.train()
            inp = sample.inp
            lbl = sample.lbl

            model.zero_grad()
            loss = model.neg_logllihood(inp, lbl)
            loss.backward()
            optim.step()

            progress_bar(i/len(train_iter), loss.item(), epoch)

        valid(model, valid_iter, opt)

        if (epoch + 1) % opt.save_per == 0:
            basename = "{}-epoch-{}".format(opt.name, epoch)
            model_fname = basename + ".model"
            torch.save(model.state_dict(), model_fname)

def valid(model, valid_iter, opt):

    model.eval()
    tags_res = []
    res = []
    y_pred = []
    y_true = []
    for i, sample in enumerate(valid_iter):
        inp = sample.inp
        # lbl: (seq_len, bsz)
        lbl = sample.lbl
        score, pred_lst = model(inp)

        # pred_batch: (bsz, seq_len)
        pred_batch = torch.cat(pred_lst, dim=1).transpose(0, 1)
        # lbl_batch: (bsz, seq_len)
        lbl_batch = lbl.transpose(0, 1)

        for pred, lbl in zip(pred_batch, lbl_batch):
            tags_t = []
            for idx_pred, idx_lbl in zip(pred, lbl):
                if idx_lbl in [model.tag2idx[TAG_EOS], model.tag2idx[PAD_TAG]]:
                    break
                tags_t.append(model.idx2tag[idx_pred])
                y_pred.append(idx_pred)
                y_true.append(idx_lbl)
            tags_res.append(tags_t)

        # print('%d/%d' % (i, len(valid_iter)))

    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    print('p/r/f: %f/%f/%f\n' % (precision, recall, f1))

    for example, tags in zip(valid_iter.dataset.examples, tags_res):
        seq_res = []
        char_seq = example.inp
        for char, tag in zip(char_seq, tags):
            seq_res.append(char)
            if tag in [TAG_END, TAG_SING]:
                seq_res.append('  ')

        res.append(''.join(seq_res)+'\n')

    with open(os.path.join(RES, opt.fpred), 'w') as f:
        f.writelines(res)

    print()

    os.system('%s %s \
    %s %s > %s' %
              (os.path.join(RES, 'score'),
               os.path.join(GOLD, opt.fwords),
               os.path.join(DATA, opt.fvalid),
               os.path.join(RES, opt.fpred),
               os.path.join(RES, 'score.utf8')))

if __name__ == '__main__':
    parser = argparse.\
        ArgumentParser(description='main.py',
                       formatter_class=argparse.
                       ArgumentDefaultsHelpFormatter)

    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()

    train_iter, valid_iter, INP, LBL = preproc.get_iters(ftrain=opt.ftrain,
                                               fvalid=opt.fvalid,
                                               bsz=opt.bsz,
                                               min_freq=opt.min_freq)
    location = opt.gpu if torch.cuda.is_available() and opt.gpu != -1 else 'cpu'
    device = torch.device(location)

    model = nets.BiLSTM_CRF(voc_size=len(INP.vocab.stoi),
                    idx2tag=LBL.vocab.itos,
                    tag2idx=LBL.vocab.stoi,
                    emb_dim=opt.emb_dim,
                    hdim=opt.hdim).to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)
    train(model, {'train': train_iter,
                  'valid': valid_iter},
          opt,
          optimizer)
