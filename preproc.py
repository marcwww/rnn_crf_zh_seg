import torchtext
from torchtext import data
from torchtext import datasets
from macros import *
import os
import io
import crash_on_ipy
import torch

def remove_blank(path):
    lines = []
    with open(path, 'r') as f:
        for line in f:
            if len(line.strip())>0:
                lines.append(line)

    with open(path, 'w') as f:
        f.writelines(lines)

def get_dataset(path, field_inp, field_lbl):
    train = data.TabularDataset(path=path,
                                format='tsv',
                                fields=[('inp', field_inp)])

    train_lbl = data.TabularDataset(path=path,
                                    format='tsv',
                                    fields=[('lbl', field_lbl)])

    for i, example in enumerate(train_lbl.examples):
        train.examples[i].lbl = example.lbl

    train.fields['lbl'] = train_lbl.fields['lbl']

    return train

def get_iters(ftrain='train.utf8',
              fvalid='valid.utf8',
              bsz=64,
              min_freq=1,
              device=-1):

    def tokenizer_input(txt):
        res = ''.join(txt.strip().split())
        return list(res)

    def tokenizer_label(txt):
        res = []
        for word in txt.strip().split():
            if len(word) == 1:
                res.append(TAG_SING)
            elif len(word) == 2:
                res.extend([TAG_BEGIN, TAG_END])
            else:
                res.extend([TAG_BEGIN] +
                           [TAG_MIDDLE] * (len(word)-2) +
                           [TAG_END])

        # res = [TAG_BOS] + res + [TAG_EOS]
        res = res + [TAG_EOS]
        return res

    remove_blank(path=os.path.join(DATA, ftrain))
    remove_blank(path=os.path.join(DATA, fvalid))

    INP = torchtext.data.Field(sequential=True,
                                tokenize=tokenizer_input,
                                pad_token=PAD_CHAR,
                                unk_token=UNK_WORD)

    LBL = torchtext.data.Field(sequential=True,
                                 tokenize=tokenizer_label,
                                 pad_token=PAD_TAG)

    train = get_dataset(os.path.join(DATA, ftrain), INP, LBL)

    INP.build_vocab(train, min_freq = min_freq)
    print('vocab size:', len(INP.vocab.itos))
    LBL.build_vocab(train)

    valid = get_dataset(os.path.join(DATA, fvalid), INP, LBL)

    train_iter = data.Iterator(train, batch_size=bsz, sort=False, repeat=False,
                               device=device)
    valid_iter = data.Iterator(valid, batch_size=bsz, sort=False, repeat=False,
                               train=False, shuffle=False, device=device)

    return train_iter, valid_iter, INP, LBL

if __name__ == '__main__':
    train_iter = get_iters('pku_training.utf8')
    for sample in train_iter:
        inp = sample.inp
        lbl = sample.lbl
        print(inp, lbl)