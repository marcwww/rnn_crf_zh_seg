import argparse

def model_opts(parser):
    group = parser.add_argument_group('model')
    group.add_argument('-emb_dim', type=int, default=30)
    group.add_argument('-hdim', type=int, default=30)

def train_opts(parser):
    group = parser.add_argument_group('train')
    group.add_argument('-ftrain', type=str, default='train_pku.utf8')
    group.add_argument('-fvalid', type=str, default='valid_pku.utf8')
    group.add_argument('-bsz', type=int, default=64)
    group.add_argument('-min_freq', type=int, default=1)
    group.add_argument('-nepoch', type=int, default=10)
    group.add_argument('-fpred', type=str, default='test_segmentation.utf8')
    group.add_argument('-fwords', type=str, default='cityu_training_words.utf8')
    group.add_argument('-save_per', type=int, default=5)
    group.add_argument('-name', type=str, default='blstm_crf')
    group.add_argument('-gpu', type=int, default=-1)
