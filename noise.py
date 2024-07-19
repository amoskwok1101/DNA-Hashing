import torch
import random
import numpy as np

def old_word_drop(vocab, x, p):     # drop words with probability p
    x_ = []
    num_drop = round((x.size(0) - 1) * p) # do not drop the start sentence symbol
    for i in range(x.size(1)):
        words = x[:, i].tolist()
        indices = random.sample(range(1, len(words)), num_drop)
        indices = set(indices)
        sent = [w for j, w in enumerate(words) if j not in indices]
        sent += [vocab.pad] * (len(words)-len(sent))
        x_.append(sent)
    return torch.LongTensor(x_).t().contiguous().to(x.device)

def word_drop(vocab, x, p):
    seq_len, batch_size = x.shape
    device = x.device
    
    # Calculate the number of words to drop in each sequence
    num_drop = round((seq_len - 1) * p)
    
    # Generate random indices to drop, but do not drop the start sentence symbol (index 0)
    drop_indices = torch.tensor([
        [random.sample(range(1, seq_len), num_drop) for _ in range(batch_size)]
    ], device=device).view(batch_size, num_drop)
    
    # Create a mask for elements to keep 
    mask = torch.permute(torch.ones_like(x, dtype=torch.bool),(1,0)) # shape (seq_len, batch_size)
    mask[torch.arange(batch_size).unsqueeze(1), drop_indices] = False
    mask = torch.permute(mask,(1,0))
    # for batch_idx in range(batch_size):
    #     mask[drop_indices[batch_idx], batch_idx] = False
        
    # Apply mask and pad sequences
    x_masked = x.t()[mask.t()].view(batch_size, seq_len - num_drop)
    x_masked = x_masked.t()
    # remark
    # reason of using x.t()[mask.t()] and .view(batch_size, seq_len - num_drop) and then transpose back:
    # x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # mask = torch.tensor([[True, False, True], [False, True, True]])
    # result = x[mask]
    pad_tensor = torch.full((num_drop, batch_size), vocab.pad, device=device)
    x_dropped = torch.cat([x_masked, pad_tensor], dim=0)
    
    return x_dropped

def old_word_add(vocab, x, p):     # add words with probability p
    x_ = []
    num_add = round((x.size(0) - 1) * p) # do not add before the start sentence symbol
    for i in range(x.size(1)):
        words = x[:, i].tolist()
        #the added position must be before len(words)-num_add
        #such that the additions will be kept after the trimming to original length
        indices = random.sample(range(1, len(words)-num_add), num_add) 
        ran_num = np.random.randint(vocab.nspecial, vocab.size, num_add)
        for i, index in enumerate(sorted(indices, reverse=True)):
            words.insert(index, ran_num[i])
        sent = words[0:x.size(0)]
        x_.append(sent)
    return torch.LongTensor(x_).t().contiguous().to(x.device)

def word_add(vocab, x, p):
    seq_len, batch_size = x.shape
    device = x.device

    num_add = round((seq_len - 1) * p)
    
    # the added position must be before len(words)-num_add 
    # such that the additions will be kept after the trimming to original length
    # shape [batch_size, num_add]
    add_indices = torch.tensor([
        [random.sample(range(1, seq_len-1-num_add), num_add) for _ in range(batch_size)]  
    ], device=device).view(batch_size, num_add)
    
    # Generate random words to insert (ensure these are within vocab size excluding special tokens)
    random_words = torch.randint(vocab.nspecial, vocab.size, (batch_size, num_add), device=device)
    
    # Create an expanded tensor
    # shape [seq_len + num_add, batch_size]
    expanded_x = torch.full((seq_len + num_add, batch_size), vocab.pad, device=device)  # Use any placeholder which will never be shown after adding words

    # Copy elements from the original x
    for i in range(seq_len):
        # Count how many insertions occur before index i 
        mask = (add_indices <= i).sum(1)  # shape [batch_size] 
        target_indices = i + mask 
        expanded_x[target_indices, torch.arange(batch_size)] = x[i]

    # This is important! 
    # For example, 
    # adding words at indices 1, 2, 3, 4, 5 --> add_indices should become 1+0, 2+1, 3+2, 4+3, 5+4 
    # (equivalent to adding its sorted position)
    add_indices += torch.argsort(torch.argsort(add_indices, dim=1), dim=1)
    
    # Insert new elements
    # for b in range(batch_size):
    #     expanded_x[add_indices[b], b] = random_words[b]
    expanded_x = torch.permute(expanded_x,(1,0))
    expanded_x[torch.arange(batch_size).unsqueeze(1),add_indices] = random_words
    expanded_x = torch.permute(expanded_x,(1,0))
    
    # trim the expanded tensor to the original sequence length [seq_len, batch_size]
    expanded_x = expanded_x[:seq_len, :]

    return expanded_x

def old_word_substitute(vocab, x, p):     # substitute words with probability p
    x_ = []
    alphabet = range(vocab.nspecial, vocab.size)
    num_substitute = round((x.size(0) - 1) * p) # do not substitute the start sentence symbol
    for i in range(x.size(1)):
        words = x[:, i].tolist()
        indices = random.sample(range(1, len(words)), num_substitute)
        for i, index in enumerate(sorted(indices)):
            words[index] = random.choice([num for num in alphabet if num != words[index]])
        x_.append(words)
    return torch.LongTensor(x_).t().contiguous().to(x.device)

def word_substitute(vocab, x, p):
    # substitute words with probability p
    seq_len, batch_size = x.shape
    device = x.device

    num_substitute = round((seq_len - 1) * p) # do not substitute the start sentence symbol

    substitute_indices = torch.tensor([
        [random.sample(range(1, seq_len), num_substitute) for _ in range(batch_size)]  
    ], device=device).view(batch_size, num_substitute)

    # Generate random words for substitution
    random_words = torch.randint(vocab.nspecial, vocab.size, (batch_size, num_substitute), device=device)
    # print(random_words)
    
    # Copy the original tensor 
    # Otherwise, the original tensor will be modified
    x_ = torch.permute(x.clone(),(1,0))
    
    # at substitute_indices, replace the words with random_words
    # for b in range(batch_size):
    #     x_[substitute_indices[b], b] = random_words[b]
    x_[torch.arange(batch_size).unsqueeze(1), substitute_indices] = random_words
    return torch.permute(x_,(1,0))

def word_drop_add_substitute(vocab, x, p):     # drop/add/substitute words with probability p
    x_ = []
    alphabet = range(vocab.nspecial, vocab.size)
    num_mutation = round((x.size(0) - 1) * p) # do not change the start sentence symbol
    for i in range(x.size(1)):
        char_list = x[:, i].tolist()
        #the added position must be before len(words)-num_mutation
        #such that the additions will be kept after the trimming to original length
        indices = random.sample(range(1, len(char_list)-num_mutation), num_mutation)
        for i, index in enumerate(sorted(indices, reverse=True)):
            mutation_type = random.choice(range(3))
            if mutation_type == 0: #drop
                char_list.pop(index)
            elif mutation_type == 1: #add
                char_list.insert(index, random.choice(alphabet))
            else: #substitute
                char_list[index] = random.choice([num for num in alphabet if num != char_list[index]])
        if len(char_list) >= x.size(0):
            sent = char_list[0:x.size(0)]
        else:
            sent = char_list[:]
            sent += [vocab.pad] * (x.size(0)-len(sent))
        x_.append(sent)
    return torch.LongTensor(x_).t().contiguous().to(x.device)

def noisy(vocab, x, drop_prob, add_prob, sub_prob, any_prob):
    if drop_prob > 0:
        x = word_drop(vocab, x, drop_prob)
    if add_prob > 0:
        x = word_add(vocab, x, add_prob)
    if sub_prob > 0:
        x = word_substitute(vocab, x, sub_prob)
    if any_prob > 0:
        x = word_drop_add_substitute(vocab, x, any_prob)
    return x

if __name__ == "__main__":
    
    import time

    from utils import set_seed, logging, load_sent, linear_decay_scheduler
    from vocab_less import Vocab
    from batchify_less import get_batches
    
    vocab = Vocab()
    
    def print_tensor(data):
        data = data.cpu().numpy().tolist()
        data = [str(d) for d in data]
        print("".join(data))
        
    train_sents = load_sent("/home/amos/MgDB/data/virus_all.fasta_noAmbiguous_cdhit_onebase_10percent_train")
    train_batches = get_batches(train_sents, vocab, 2048, "cuda")
    old_drop_time = []
    drop_time = []
    old_add_time = []
    add_time = []
    old_sub_time = []
    sub_time = []
    old_mix_time = []
    mix_time = []
    # print(len(train_batches))
    for batch in train_batches:
        
        start = time.time()
        new = word_add(vocab, batch, 0.05)
        end = time.time()
        # print("new time: ", end-start)
        sub_time.append(end-start)
        
        start = time.time()
        # print(noisy(vocab, batch, 0.2, 0, 0, 0))
        old = old_word_add(vocab, batch, 0.05)
        end = time.time()
        # print("old time: ", end-start)
        old_sub_time.append(end-start)
        
        # print_tensor([batch[:, i] for i in range(batch.size(1))][0])
        # print_tensor([old[:, i] for i in range(old.size(1))][0])
        # print_tensor([new[:, i] for i in range(new.size(1))][0])
        
        # start = time.time()
        # print(noisy(vocab, batch, 0, 0.2, 0, 0))
        # end = time.time()
        # print("time: ", end-start)
        # add_time.append(end-start)
        
        # start = time.time()
        # print(noisy(vocab, batch, 0, 0, 0.2, 0))
        # end = time.time()
        # print("time: ", end-start)
        # sub_time.append(end-start)

        # start = time.time()
        # print(noisy(vocab, batch, 0, 0, 0, 0.2))
        # end = time.time()
        # print("time: ", end-start)
        # mix_time.append(end-start)
        
    print("mean time: ", np.mean(old_sub_time))
    print("mean time: ", np.mean(sub_time))
    # print(sub_time)
    # print(old_sub_time)