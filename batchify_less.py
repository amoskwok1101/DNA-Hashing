import torch

def get_batch(x, vocab, device):
    # go_x, x_eos = [], []
    go_x = []
    max_len = max([len(s) for s in x])
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        # x_eos.append(s_idx + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous() # time * batch

# def get_batches(data, vocab, batch_size, device):
#     order = range(len(data))
#     z = sorted(zip(order, data), key=lambda i: len(i[1]))
#     order, data = zip(*z)

#     batches = []
#     i = 0
#     while i < len(data):
#         j = i
#         while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
#             j += 1
#         batches.append(get_batch(data[i: j], vocab, device))
#         i = j
#     return batches, None

from itertools import groupby

def get_batches(data, vocab, batch_size, device):
    # Group the data by the length of the sequences
    data = sorted(data, key=len)
    grouped_data = groupby(data, key=len)

    batches = []
    for _, group in grouped_data:
        group_list = list(group)
        for i in range(0, len(group_list), batch_size):
            batch_data = group_list[i:i+batch_size]
            batches.append(get_batch(batch_data, vocab, device))

    return batches