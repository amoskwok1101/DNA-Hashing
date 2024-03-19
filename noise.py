import numpy as np
import torch
import random
import time

def word_shuffle(vocab, x, k):   # slight shuffle such that |sigma[i]-i| <= k
    base = torch.arange(x.size(0), dtype=torch.float).repeat(x.size(1), 1).t()
    inc = (k+1) * torch.rand(x.size())
    inc[x == vocab.go] = 0     # do not shuffle the start sentence symbol
    inc[x == vocab.pad] = k+1  # do not shuffle end paddings
    _, sigma = (base + inc).sort(dim=0)
    return x[sigma, torch.arange(x.size(1))]

def word_drop(vocab, x, p):     # drop words with probability p
    x_ = []
    for i in range(x.size(1)):
        words = x[:, i].tolist()
        keep = np.random.rand(len(words)) > p
        keep[0] = True  # do not drop the start sentence symbol
        sent = [w for j, w in enumerate(words) if keep[j]]
        sent += [vocab.pad] * (len(words)-len(sent))
        x_.append(sent)
    return torch.LongTensor(x_).t().contiguous().to(x.device)

def word_add(vocab, x, p):     # add words with probability p
    x_ = []
    for i in range(x.size(1)):
        words = x[:, i].tolist()
        num_add = round((len(words) - 1) * p) # do not add before the start sentence symbol
        indices = random.sample(range(1, len(words)), num_add)
        ran_num = np.random.randint(vocab.nspecial, vocab.size, num_add)
        for i, index in enumerate(sorted(indices, reverse=True)):
            words.insert(index, ran_num[i])
        sent = words[0:x.size(0)]
        x_.append(sent)
    return torch.LongTensor(x_).t().contiguous().to(x.device)

def word_blank(vocab, x, p):     # blank words with probability p
    blank = (torch.rand(x.size(), device=x.device) < p) & \
        (x != vocab.go) & (x != vocab.pad)
    x_ = x.clone()
    x_[blank] = vocab.blank
    return x_

def word_substitute(vocab, x, p):     # substitute words with probability p
    keep = (torch.rand(x.size(), device=x.device) > p) | \
        (x == vocab.go) | (x == vocab.pad)
    x_ = x.clone()
    x_.random_(vocab.nspecial, vocab.size)
    x_[keep] = x[keep]
    return x_

def word_drop_add_substitute(vocab, x, p):     # drop/add/substitute words with probability p
    x_ = []
    for i in range(x.size(1)):
        char_list = x[:, i].tolist()
        num = round((len(char_list) - 1) * p) # do not change at the start sentence symbol
        indices = random.sample(range(1, len(char_list)), num)
        ran_num = np.random.randint(vocab.nspecial, vocab.size, num)
        for i, index in enumerate(sorted(indices, reverse=True)):
            mutation_type = random.choice(range(3))
            if mutation_type == 0: #drop
                char_list.pop(index)
            elif mutation_type == 1: #add
                char_list.insert(index, ran_num[i])
            else: #substitute
                char_list[index] = ran_num[i]
        if len(char_list) >= x.size(0):
            sent = char_list[0:x.size(0)]
        else:
            sent = char_list[:]
            sent += [vocab.pad] * (x.size(0)-len(sent))
        x_.append(sent)
    return torch.LongTensor(x_).t().contiguous().to(x.device)

def noisy(vocab, x, drop_prob, add_prob, sub_prob, any_prob):
    #if shuffle_dist > 0:
    #    x = word_shuffle(vocab, x, shuffle_dist)
    if add_prob > 0:
        x = word_add(vocab, x, add_prob)
    if drop_prob > 0:
        x = word_drop(vocab, x, drop_prob)
    if sub_prob > 0:
        x = word_substitute(vocab, x, sub_prob)
    if any_prob > 0:
        x = word_drop_add_substitute(vocab, x, any_prob)
    return x

nucleotides = ['A', 'C', 'G', 'T']

def word_drop_dna(x, p):     # drop words with probability p, deletion
    x_ = []
    for s in x:
        char_list = list(s)
        num_drop = round(len(char_list) * p)
        indices = random.sample(range(len(char_list)), num_drop)
        #reverse sort, will not change the undo index
        for index in sorted(indices, reverse=True):
            char_list.pop(index)
        #x_.append(' '.join(char_list))
        x_.append(char_list)
    return x_

def word_add_dna(x, p):     # add words with probability p, insertion
    x_ = []
    for s in x:
        char_list = list(s)
        num_add = round(len(char_list) * p)
        indices = random.sample(range(len(char_list)), num_add)
        for index in sorted(indices, reverse=True):
            char_list.insert(index, random.choice(nucleotides))
        if len(char_list) >= x.size(0):
            char_list = char_list[0:x.size(0)]
        x_.append(char_list)
    return x_

def word_substitute_dna(x, p):     # substitute words with probability p, mutation
    x_ = []
    for s in x:
        char_list = list(s)
        num_substitute = round(len(char_list) * p)
        indices = random.sample(range(len(char_list)), num_substitute)
        for index in indices:
            bases = nucleotides[:]
            bases.remove(char_list[index])
            char_list[index] = random.choice(bases)
        x_.append(char_list)
    return x_

def word_drop_add_substitute_item_dna(s, p):     # drop/add/substitute words with probability p for one sent
    char_list = list(s)
    num = round(len(char_list) * p)
    indices = random.sample(range(len(char_list)), num)
    for index in sorted(indices, reverse=True):
        mutation_type = random.choice(range(3))
        if mutation_type == 0:
            char_list.pop(index)
        elif mutation_type == 1:
            char_list.insert(index, random.choice(nucleotides))
        else:
            bases = nucleotides[:]
            bases.remove(char_list[index])
            char_list[index] = random.choice(bases)    
    return char_list

def word_drop_add_substitute_dna(x, p):     # drop/add/substitute words with probability p
    x_ = []
    for s in x:
        char_list = list(s)
        num = round(len(char_list) * p)
        indices = random.sample(range(len(char_list)), num)
        for index in sorted(indices, reverse=True):
            mutation_type = random.choice(range(3))
            if mutation_type == 0:
                char_list.pop(index)
            elif mutation_type == 1:
                char_list.insert(index, random.choice(nucleotides))
            else:
                bases = nucleotides[:]
                bases.remove(char_list[index])
                char_list[index] = random.choice(bases)

        if len(char_list) >= x.size(0):
            char_list = char_list[0:x.size(0)]
        x_.append(char_list)
    return x_

def noisy_dna(x, drop_prob, add_prob, sub_prob, any_prob):
    if drop_prob > 0:
        x = word_drop_dna(x, drop_prob)
    if add_prob > 0:
        x = word_add_dna(x, add_prob)
    if sub_prob > 0:
        x = word_substitute_dna(x, sub_prob)
    if any_prob > 0:
        x = word_drop_add_substitute_dna(x, any_prob)
    return x