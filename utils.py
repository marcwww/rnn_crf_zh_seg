import torch

def log_sum_exp(vec):
    # vec: (bsz, tag_size)
    # max_score: (bsz, 1)
    max_score = torch.max(vec, dim=1)[0].unsqueeze(-1)
    # s_max + log(sum(exp(vec-s_max)))
    # = s_max + log(sum(exp(vec)/exp(s_max)))
    # = s_max + log(1/exp(s_max)\cdot sum(exp(vec)))
    # = s_max + -s_max + log(sum(exp(vec)))
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score)))

def argmax(vec):
    # vec: (bsz, tag_size)
    _, idx = torch.max(vec, 1)
    return idx