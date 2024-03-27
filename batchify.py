import torch

def get_batch(x, vocab, device):
    go_x, x_eos = [], []
    max_len = max([len(s) for s in x])
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch

#ATCG -> [A T C G]
def get_batch_dna(x, vocab, device):
    go_x = []
    for s in x:
        s_idx = [w for w in s[0]]
        go_x.append(s_idx)
    return go_x

def get_batches(data, vocab, batch_size, device):
    order = range(len(data))
    z = sorted(zip(order, data), key=lambda i: len(i[1]))
    order, data = zip(*z)

    batches = []
    i = 0
    while i < len(data):
        j = i
        while j < min(len(data), i+batch_size) and len(data[j]) == len(data[i]):
            j += 1
        batches.append(get_batch_dna(data[i: j], vocab, device))
        i = j
    return batches, order

def tokenization(x, vocab, device, max_len):
    go_x, x_eos = [], []
    for s in x:
        s_idx = [vocab.word2idx[w] if w in vocab.word2idx else vocab.unk for w in s]
        padding = [vocab.pad] * (max_len - len(s))
        go_x.append([vocab.go] + s_idx + padding)
        x_eos.append(s_idx + [vocab.eos] + padding)
    return torch.LongTensor(go_x).t().contiguous().to(device), \
           torch.LongTensor(x_eos).t().contiguous().to(device)  # time * batch

#if k = 2, [A T C G] - > [AT TC CG]
def convert_sequences_to_kmers(sequences, k):
    kmer_sequences=[]
    for sequence in sequences:
        sequence = ''.join(sequence)
        kmers = []
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmers.append(kmer)
        kmer_sequences.append(kmers)
    return kmer_sequences

    
