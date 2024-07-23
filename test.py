import argparse
import os
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math

# from vocab import Vocab
from model import *
from utils import *
from batchify import get_batches
from train import evaluate
# from noise import noisy_dna
from meter import AverageMeter
import pandas as pd
from tqdm import tqdm
import pickle
import torch.nn as nn
from annoy import AnnoyIndex

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', metavar='DIR', required=True,
                    help='checkpoint directory')
parser.add_argument('--output', metavar='FILE',
                    help='output file name (in checkpoint directory)')
parser.add_argument('--data', metavar='FILE',
                    help='path to data file')

parser.add_argument('--enc', default='mu', metavar='M',
                    choices=['mu', 'z'],
                    help='encode to mean of q(z|x) or sample z from q(z|x)')
parser.add_argument('--dec', default='greedy', metavar='M',
                    choices=['greedy', 'sample'],
                    help='decoding algorithm')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--max-len', type=int, default=35, metavar='N',
                    help='max sequence length')

parser.add_argument('--evaluate', action='store_true',
                    help='evaluate on data file')
parser.add_argument('--triplet', action='store_true',
                    help='evaluate the true positve for triplet loss')
parser.add_argument('--cutoff', type=float, default=0.01, metavar='N',
                    help='A true positive if triplet loss is less than cutoff')
parser.add_argument('--ppl', action='store_true',
                    help='compute ppl by importance sampling')
parser.add_argument('--reconstruct', action='store_true',
                    help='reconstruct data file')
parser.add_argument('--sample', action='store_true',
                    help='sample sentences from prior')

parser.add_argument('--interpolate', action='store_true',
                    help='interpolate between pairs of sentences')
parser.add_argument('--latent-nn', action='store_true',
                    help='find nearest neighbor of sentences in the latent space')
parser.add_argument('--m', type=int, default=100, metavar='N',
                    help='num of samples for importance sampling estimate')
parser.add_argument('--n', type=int, default=5, metavar='N',
                    help='num of sentences to generate for sample/interpolate')
parser.add_argument('--k', type=float, default=1, metavar='R',
                    help='k * offset for vector arithmetic')
parser.add_argument('--metrics', action='store_true',
                    help='evaluate on data file by metrics')
parser.add_argument('--metrics_hm', action='store_true',
                    help='evaluate on data file by metrics in hamming space')
parser.add_argument('--distance_type', default='euclidean', metavar='M',
                    choices=['cosine', 'euclidean', 'hamming'],
                    help='which distance is to used in the metrics')
parser.add_argument('--noise', default='0,0,0,0', metavar='P,P,P,P',
                    help='word drop prob, add_prob, substitute prob, any_prob')

# Architecture
parser.add_argument('--use-transformer', action='store_true',
                    help='whether to use transformer model')
parser.add_argument('--no-Attention', action='store_true', default=True,
                    help='indicate to use attention mechanism')
parser.add_argument('--use-less', action='store_true',
                    help='use simplified version code')

# Setting
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')

# Mode
parser.add_argument('--generate', action='store_true',
                    help='generate noisy DNA sequences')
parser.add_argument('--embedding', action='store_true',
                    help='calculate embeddings')
parser.add_argument('--generate-sequential', action='store_true',
                    help='generate 4 noisy DNA sequences sequentially for each original DNA sequence')
parser.add_argument('--metrics-sequential', action='store_true',
                    help='generate 4 noisy DNA sequences and calculate distance sequentially for each original DNA sequence')
parser.add_argument('--noisy-levels', default='0.05,0.1,0.15,0.2', metavar='P,P,P,P',
                    help='4 noisy levels')
parser.add_argument('--embedding-all', action='store_true',
                    help='calculate embeddings for all data (no noise)')
parser.add_argument('--test-retrieval', action='store_true',
                    help='test retrieval accuracy')
parser.add_argument('--custom-generate', action='store_true',
                    help='generate custom number of noisy DNA sequences')
parser.add_argument('--custom-noisy-num', type=int, default=10, metavar='N',
                    help='number of noisy DNA sequences to generate')
parser.add_argument('--specific-model-file', metavar='FILE', 
                    help='specific model file to load')
parser.add_argument('--use-annoy', action='store_true',
                    help='use annoy for retrieval')

parser.add_argument('--use-median', action='store_true',
                    help='use 32 median for toBinary cutoff')
parser.add_argument('--median-file', metavar='FILE',
                    help='file path of the median')
parser.add_argument('--retrieval-report', default='', 
                    help='file path of the retrieval report')
parser.add_argument('--retrieval-by-radius', action='store_true',
                    help='1. compute the radius (hamming distance between anchor and sample), \
                    2. retrieval all sequences in db whose hamming distance with the anchor is less than or equal to the radius')
def get_model(path):
    ckpt = torch.load(path)
    train_args = ckpt['args']
    train_args.use_less = True if args.use_less else False
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[train_args.model_type](
        vocab, train_args).to('cuda')
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model, train_args
def create_average_meter():
    return AverageMeter()
"""
def encode(sents):
    assert args.enc == 'mu' or args.enc == 'z'
    batches, order = get_batches(sents, vocab, args.batch_size, device)
    z = []
    for inputs, _ in batches:
        print(inputs.shape)
        if args.enc == 'mu':
            _, _, zi = model.encode2z(inputs, is_test=True)
        else:
            _, _, zi = model.encode2z(inputs)
        z.append(zi.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z
    return z_
"""

def encode(inputs):
    assert args.enc == 'mu' or args.enc == 'z'
    if args.enc == 'mu':
        _, _, z = model.encode2z(inputs, is_test=True)
    else:
        _, _, z = model.encode2z(inputs)
    return z

"""
def encode_hm(sents):
    batches, order = get_batches(sents, vocab, args.batch_size, device)
    z = []
    for inputs, _ in batches:
        _, _, zi = model.encode2binary(inputs, is_test=True)
        z.append(zi.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z
    return z_
"""
def encode_hm(inputs):
    assert args.enc == 'mu' or args.enc == 'z'
    if args.enc == 'mu':
        _, _, z = model.encode2binary(inputs, is_test=True)
    else:
        _, _, z = model.encode2binary(inputs)
    return z

def decode(z):
    sents = []
    i = 0
    while i < len(z):
        zi = torch.tensor(z[i: i+args.batch_size], device=device)
        outputs = model.generate(zi, args.max_len, args.dec).t()
        for s in outputs:
            sents.append([vocab.idx2word[id] for id in s[1:]])  # skip <go>
        i += args.batch_size
    return strip_eos(sents)

def calc_ppl(sents, m):
    batches, _ = get_batches(sents, vocab, args.batch_size, device)
    total_nll = 0
    with torch.no_grad():
        for inputs, targets in batches:
            total_nll += model.nll_is(inputs, targets, m).sum().item()
    n_words = sum(len(s) + 1 for s in sents)    # include <eos>
    return total_nll / len(sents), np.exp(total_nll / n_words)

def euclidean_distance(x, y, dim_z):
    # x = nn.Tanh()(torch.tensor(x)).numpy()
    # y = nn.Tanh()(torch.tensor(y)).numpy()
    return np.sqrt(np.sum(np.power(x - y, 2), axis = 0)) / dim_z

def cosine_distance(x, y):
    # x = nn.Tanh()(torch.tensor(x)).numpy()
    # y = nn.Tanh()(torch.tensor(y)).numpy()
    cos_sim = x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return 1 - cos_sim

def hamming_distance(x, y, dim_z):
    x = toBinary(x)
    y = toBinary(y)
    return np.sum(np.abs(x - y), axis = 0) / dim_z

def hamming_distance_bi(x, y, dim_z):
    x = toBinary(x)
    return np.sum(np.abs(x - y), axis = 0) / dim_z

# def sigmoid(x):
#     #return 1/(1 + np.exp(-x))
#     l = len(x)
#     y = []
#     for i in range(l):
#         if x[i] >= 0:
#             y.append(1.0/(1+np.exp(-x[i])))
#         else :
#             y.append(np.exp(x[i])/(np.exp(x[i])+1))
#     return np.array(y)
# File "test.py", line 206, in sigmoid
# if x[i] >= 0:
# ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

def sigmoid(x):
    """Compute the sigmoid function in a vectorized way."""
    return nn.Sigmoid()(torch.tensor(x)).numpy()


# def toBinary(code):
#     code = sigmoid(code)
#     code = code - 0.5
#     code = (np.sign(code) + 1) / 2
#     return code

def toBinary(code):
    # code = sigmoid(code)
    # if args.use_median and args.median_file is not None:
    #     median = pickle.load(open(args.median_file, 'rb'))
    #     binary_code = (code > median).astype(int)  # Using threshold of median
    # else:
    #     binary_code = (code > 0.5).astype(int) # Using threshold of 0.5
    binary_code = (code > 0).astype(int)
    return binary_code

def evaluate_triplet(model, batches, cutoff = 0.01):
    model.eval()
    TP = 0
    count = 0
    with torch.no_grad():
        for inputs, targets in batches:
            losses = model.autoenc(inputs, targets, is_test=True, is_evaluate=True)
            triplet_loss = losses['triplet']
            triplet_loss = torch.where(triplet_loss < cutoff, 1, 0)
            TP += torch.sum(triplet_loss, dim=0)
            count += triplet_loss.size(0)
    return TP, count

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_less:
        from vocab_less import Vocab
        vocab = Vocab()
    else:
        from vocab_less import Vocab
        vocab = Vocab()

    if args.seed:
        set_custom_seed(args.seed)
    else:
        set_seed()
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if args.specific_model_file:
        model, train_args = get_model(args.specific_model_file)
    else:
        model, train_args = get_model(os.path.join(args.checkpoint, 'model.pt'))

    if args.evaluate:
        sents = load_sent(args.data)
        batches, _ = get_batches(sents, vocab, args.batch_size, device)
        meters = evaluate(model, batches)
        print(' '.join(['{} {:.4f},'.format(k, meter.avg)
            for k, meter in meters.items()]))

    if args.ppl:
        sents = load_sent(args.data)
        nll, ppl = calc_ppl(sents, args.m)
        print('NLL {:.2f}, PPL {:.2f}'.format(nll, ppl))

    if args.sample:
        z = np.random.normal(size=(args.n, model.args.dim_z)).astype('f')
        sents = decode(z)
        write_sent(sents, os.path.join(args.checkpoint, args.output))

    if args.reconstruct:
        sents = load_sent(args.data)
        z = encode(sents)
        sents_rec = decode(z)
        write_z(z, os.path.join(args.checkpoint, args.output+'.z'))
        write_sent(sents_rec, os.path.join(args.checkpoint, args.output+'.rec'))

    if args.embedding:
        import pickle
        args.noise = [float(x) for x in args.noise.split(',')]
        sents = load_sent(args.data)
        num_sents = len(sents)
        batches, order = get_batches(sents, vocab, args.batch_size, device)
        anchor = []
        generate_sample = []
        for inputs, _ in batches:
            inputs_noisy = noisy(vocab, inputs, *args.noise)
            zi_anchor = encode(inputs)
            zi_sample = encode(inputs_noisy)
            anchor.append(zi_anchor.detach().cpu().numpy())
            generate_sample.append(zi_sample.detach().cpu().numpy())
        
        anchor = np.concatenate(anchor, axis=0)
        z_ = np.zeros_like(anchor)
        z_[np.array(order)] = anchor
        anchor = z_

        generate_sample = np.concatenate(generate_sample, axis=0)
        z_ = np.zeros_like(generate_sample)
        z_[np.array(order)] = generate_sample
        generate_sample = z_

        hundred = math.floor(num_sents/100)
        all_embeddings_group = []
        for i in range(hundred):
            embeddings_in_one_group = []
            embeddings_in_one_group.append(anchor[i*100])
            for j in range(100):
                embeddings_in_one_group.append(generate_sample[i*100+j])
            all_embeddings_group.append(embeddings_in_one_group)

        # print(len(all_embeddings_group))
        # print(len(all_embeddings_group[0]))
        # print(all_embeddings_group[0][0].shape)
        # for i in all_embeddings_group:
        #     for j in i:
        #         print(j)
        
        with open('result/all_embeddings'+f'_{args.noise}'+'.pkl', 'wb') as f:
            pickle.dump(all_embeddings_group, f)


    def print_list(input_list, end = '\n'):
        print(' '.join(map(str, input_list)), end = end)

    if args.generate:
        args.noise = [float(x) for x in args.noise.split(',')]
        sents = load_sent(args.data)
        num_sents = len(sents)
        batches, order = get_batches(sents, vocab, args.batch_size, device)
        anchor = []
        generate_sample = []
        for inputs, _ in batches:
            inputs_noisy = noisy(vocab, inputs, *args.noise)
            inputs = inputs[:,0].tolist()
            print("Original")
            print_list([vocab.idx2word[id] for id in inputs[1:]])
            print("Noisy")
            inputs_noisy_list = inputs_noisy.t().tolist()
            for dummy in inputs_noisy_list:
                print_list([vocab.idx2word[id] for id in dummy[1:]])

    if args.metrics:
        args.noise = [float(x) for x in args.noise.split(',')]
        sents = load_sent(args.data)
        num_sents = len(sents)
        batches, order = get_batches(sents, vocab, args.batch_size, device)
        anchor = []
        generate_sample = []
        for inputs, _ in batches:
            inputs_noisy = noisy(vocab, inputs, *args.noise)
            zi_anchor = encode(inputs)
            zi_sample = encode(inputs_noisy)
            anchor.append(zi_anchor.detach().cpu().numpy())
            generate_sample.append(zi_sample.detach().cpu().numpy())
        
        anchor = np.concatenate(anchor, axis=0)
        z_ = np.zeros_like(anchor)
        z_[np.array(order)] = anchor
        anchor = z_

        generate_sample = np.concatenate(generate_sample, axis=0)
        z_ = np.zeros_like(generate_sample)
        z_[np.array(order)] = generate_sample
        generate_sample = z_
        
        hundred = math.floor(num_sents/100)
        for i in range(hundred): # rows
            for j in range(100): # cols
                if args.distance_type == "hamming":
                    dist = hamming_distance(anchor[i*100 + j], generate_sample[i*100 + j], train_args.dim_z)
                elif args.distance_type == "cosine":
                    dist = cosine_distance(anchor[i*100 + j], generate_sample[i*100 + j]);
                elif args.distance_type == "euclidean":
                    dist = euclidean_distance(anchor[i*100 + j], generate_sample[i*100 + j], train_args.dim_z)
                print(dist, end = ',')
            print()
        
        for j in range(num_sents%100):
            if args.distance_type == "hamming":
                dist = hamming_distance(anchor[hundred*100 + j], generate_sample[hundred*100 + j], train_args.dim_z)
            elif args.distance_type == "cosine":
                dist = cosine_distance(anchor[hundred*100 + j], generate_sample[hundred*100 + j])
            elif args.distance_type == "euclidean":
                dist = euclidean_distance(anchor[hundred*100 + j], generate_sample[hundred*100 + j], train_args.dim_z)
            print(dist, end = ',')
        print()

    if args.custom_generate:
        sents = load_sent(args.data)
        num_sents = len(sents)
        batches, order = get_batches(sents, vocab, 1, device)
        for inputs, _ in batches:
            print("Original")
            print_list([vocab.idx2word[id] for id in inputs[1:]])
            print("Noisy")
            for i in range(args.custom_noisy_num):
                random_noise = random.randint(2, 20)/100 # random generate 0.02 to 0.2
                args.noise = [0, 0, 0, random_noise]
                inputs_noisy = noisy(vocab, inputs, *args.noise)
                print_list([vocab.idx2word[id] for id in inputs_noisy[1:]], end=f', {random_noise}\n')
    
    if args.generate_sequential:
        args.noisy_levels = [float(x) for x in args.noisy_levels.split(',')] # [0.05, 0.1, 0.15, 0.2]
        sents = load_sent(args.data)
        num_sents = len(sents)
        batches, order = get_batches(sents, vocab, 1, device)
        for inputs, _ in batches:
            print("Original")
            print_list([vocab.idx2word[id] for id in inputs[1:]])
            print("Noisy")
            for noise_level in args.noisy_levels:
                args.noise = [0, 0, 0, noise_level]
                inputs_noisy = noisy(vocab, inputs, *args.noise)
                print_list([vocab.idx2word[id] for id in inputs_noisy[1:]])
            
    if args.metrics_sequential:
        args.noisy_levels = [float(x) for x in args.noisy_levels.split(',')] # [0.05, 0.1, 0.15, 0.2]
        sents = load_sent(args.data)
        num_sents = len(sents)
        batches, order = get_batches(sents, vocab, 1, device)
        anchor = []
        generate_sample = []
        for inputs, _ in batches:
            zi_anchor = encode(inputs)
            anchor.append(zi_anchor.detach().cpu().numpy())
            each_generate_sample = []
            for noise_level in args.noisy_levels:
                args.noise = [0, 0, 0, noise_level]
                inputs_noisy = noisy(vocab, inputs, *args.noise)
                zi_sample = encode(inputs_noisy)
                each_generate_sample.append(zi_sample.detach().cpu().numpy())
            generate_sample.append(each_generate_sample)
        
        anchor = np.array(anchor)
        anchor = np.squeeze(anchor, axis=1)
        generate_sample = np.array(generate_sample)
        generate_sample = np.squeeze(generate_sample, axis=2)
        # print(anchor.shape)
        # print(generate_sample.shape)
        
        for i in range(anchor.shape[0]): # rows
            for j in range(4): # cols
                if args.distance_type == "hamming":
                    dist = hamming_distance(anchor[i], generate_sample[i][j], train_args.dim_z)
                elif args.distance_type == "cosine":
                    dist = cosine_distance(anchor[i], generate_sample[i][j])
                elif args.distance_type == "euclidean":
                    dist = euclidean_distance(anchor[i], generate_sample[i][j], train_args.dim_z)
                print(dist, end = ',')
            print()
    
    def list2str(list):
        return ''.join(map(str, list))
    
    def binary_array_to_int(binary_array):
        out = 0
        for bit in binary_array:
            out = (out << 1) | int(bit)
        return out
    
    def int_to_binary_array(value, length=32):
        # Generate binary string from the integer, filling with leading zeros to match the expected length
        binary_string = f'{value:0{length}b}'
        # Convert string to a list of integers, then to a NumPy array
        return np.array([int(bit) for bit in binary_string], dtype=np.uint8)
    
    def numpy_array_to_binary_string(array):
        return ''.join(str(int(x)) for x in array)

    if args.embedding_all:
        """
        Store all embeddings for all data (no noise) in a pickle file
        Format: [{'seq': 'AAT...TTA', 'embedding': array([1, 0, 0, ..., 0, 1, 0])}, ...]
        """
        args.batch_size = 512
        sents = load_sent(args.data)
        num_sents = len(sents)
        batches, order = get_batches(sents, vocab, args.batch_size, device)
        seq_embeddings = []
        float_embeddings = []
        # tqdm
        for inputs, _ in tqdm(batches, total=len(batches), desc="Generating embeddings"):
            zi = encode(inputs) # (batch_size, 32)
            zi = zi.detach().cpu().numpy()
            # zi_binary = toBinary(zi)
            # zi_binary_str = zi_binary.astype(int) # (batch_size, 32)
            # normalize zi
            # zi = F.normalize(zi, p=2, dim=1).numpy()
            zi= zi[:,:32]
            for i in range(len(zi)):
                seq_embedding = {'seq': list2str(vocab.idx2word[id] for id in inputs.t()[i][1:]), 'embedding_float':zi[i]}
                seq_embeddings.append(seq_embedding)

                float_embeddings.append(zi[i])
        
        float_embeddings = np.array(float_embeddings)
        # float_embeddings = sigmoid(float_embeddings)

        # find median of each embedding dimension
        median = np.median(float_embeddings, axis=0)
        if args.median_file is not None:
            with open(args.median_file, 'wb') as f:
                pickle.dump(median, f)
                
        # use the above just stored median file for toBinary cutoff
        for i in range(len(seq_embeddings)):
            zi_binary = toBinary(seq_embeddings[i]['embedding_float'])
            seq_embeddings[i]['embedding_bi'] = zi_binary

        with open('all_embeddings.pkl', 'wb') as f:
            pickle.dump(seq_embeddings, f)

    def hamming_distance(x, y):
        return bin(x ^ y).count('1')

    def int_to_binary_vector(x, num_bits=32):
        """ Convert an integer to a binary vector of a specified number of bits. """
        return [int(b) for b in format(x, '0{}b'.format(num_bits))]

    def build_annoy_index(embeddings, num_bits=32, n_trees=10, save_path='annoy_index.ann'):
        # Initialize Annoy index with the specified number of dimensions (bits) and metric
        annoy_metric = {'hamming': 'hamming', 'euclidean': 'euclidean', 'cosine': 'angular'}[args.distance_type]
        t = AnnoyIndex(num_bits, annoy_metric)

        for i in tqdm(range(len(embeddings)), desc="Building Annoy Index"):
            t.add_item(i, embeddings[i])

        t.build(n_trees)

        if save_path:
            t.save(save_path)
        
        return t
    
    def load_annoy_index(save_path, num_bits=32):
        t = AnnoyIndex(num_bits, 'hamming')
        t.load(save_path)
        return t

    def find_nearest_neighbors_annoy(query_embedding, annoy_index, k=1, num_bits=32):
        nearest_neighbors_indices = annoy_index.get_nns_by_vector(query_embedding, k, include_distances=False)
        return nearest_neighbors_indices
    
    # def hamming_distance_matrix(query_embedding, embeddings):
    #     query_embedding = torch.tensor(np.array(query_embedding)).cuda()
    #     embeddings = torch.tensor(np.array(embeddings)).cuda()
    #     expanded_query = query_embedding.unsqueeze(0).repeat(embeddings.size(0), 1)
    #     differences = expanded_query != embeddings
    #     distances = differences.float().sum(dim=1)
    #     return distances
    
    def hamming_distance_matrix(query_embedding, embeddings):
        query_bipolar = 2 * query_embedding - 1
        # query_embedding = torch.tensor(query_embedding, dtype=torch.float32).cuda()
        
        embeddings_bipolar = 2 *embeddings - 1
        # embeddings_bipolar = torch.tensor(embeddings, dtype=torch.float32).cuda()
        
        distances = torch.mm(query_bipolar.view(1, -1), embeddings_bipolar.t())
        
        d = query_embedding.shape[0]
        hamming_distances = (d - distances) / 2

        return hamming_distances.view(-1)
    

    # def hamming_distance_matrix(query, embeddings):
    #     # Ensure the query and embeddings are of boolean type or properly represent binary data
    #     # Broadcasting the query across all rows of embeddings and computing the difference
    #     differences = query != embeddings
        
    #     # Summing up the differences row-wise to get the Hamming distance for each embedding
    #     hamming_distances = np.sum(differences, axis=1)
        
    #     return torch.tensor(hamming_distances)
    
    from scipy.spatial import distance
    # import numpy as np
    def l2_distance_matrix(query_embedding, embeddings):
        # Compute distances using cdist
        # distances = distance.cdist([query_embedding], embeddings, 'euclidean')[0]
        return (query_embedding-embeddings).norm(p=2, dim=1)
    
    def cosine_distance_matrix(query_embedding, embeddings):
        # Compute distances using cdist
        distances = distance.cdist([query_embedding], embeddings, 'cosine')[0]
        return torch.tensor(distances, dtype=torch.float32)

    # def find_nearest_neighbors_l2(query_embedding, seq_embeddings, k=1):
    #     distances = l2_distance_matrix(query_embedding, [seq_embedding['embedding_float'] for seq_embedding in seq_embeddings])
    #     nearest_neighbors_indices = torch.topk(distances, k, largest=False).indices.cpu().numpy().tolist()
    #     return nearest_neighbors_indices
    
    # def find_nearest_neighbors_bi(query_embedding, seq_embeddings, k=1):
    #     distances = hamming_distance_matrix(query_embedding, [seq_embedding['embedding_bi'] for seq_embedding in seq_embeddings])
    #     nearest_neighbors_indices = torch.topk(distances, k, largest=False).indices.cpu().numpy().tolist()
    #     return nearest_neighbors_indices
    def find_nearest_neighbors(distances, k=1):
        nearest_neighbors_indices = torch.topk(distances, k, largest=False).indices.cpu().numpy().tolist()
        # sort distances in ascending order
        distances_sorted_dict = torch.sort(distances)
        distances_sorted = distances_sorted_dict.values.cpu().numpy()
        distances_sorted_index = distances_sorted_dict.indices.cpu().numpy()
        def get_shared_ranking(distances,k):
            ranking = []
            for i in range(len(distances)):
                if i == 0:
                    ranking.append(1)
                elif distances[i] == distances[i-1]:
                    ranking.append(ranking[-1])
                else:
                    ranking.append(len(ranking)+1)
                if ranking[-1] > k:
                    ranking = ranking[:-1]
                    break
            return np.array(ranking)
        shared_ranking = get_shared_ranking(distances_sorted,k)
        
        
        shared_nearest_neighbors_indices = distances_sorted_index[0:len(shared_ranking)]
        return nearest_neighbors_indices, shared_nearest_neighbors_indices,shared_ranking
    
    def average_of_list(numbers):
        if len(numbers) == 0:
            return 0  # Return 0 if the list is empty to avoid division by zero
        return sum(numbers) / len(numbers)
    
    if args.test_retrieval:
        args.batch_size = 256
        seq_embeddings = pickle.load(open('all_embeddings.pkl', 'rb'))
        seqs_db = np.array([seq_embedding['seq'] for seq_embedding in seq_embeddings])

        # empty the file
        if args.retrieval_report:
            with open(args.retrieval_report+'.txt', 'w') as f:
                pass

        sents = load_sent(args.data)
        num_sents = len(sents)
        batches, order = get_batches(sents, vocab, args.batch_size, device)
        noise_npercent_topm_correct_bi = {}
        noise_npercent_topm_correct_float = {}
        shared_noise_npercent_topm_correct_bi = {}
        shared_noise_npercent_topm_correct_float = {}
        match_stat = {}
        index={}
        dist={}
        hitMinusTop1_distances = {}
        for noise in [0.02, 0.05]:
            for k in [1, 10, 100]:
                index[f"{noise}_top{k}"] = []
                dist[f"{noise}_top{k}"] = []
        if args.use_annoy:
            annoy_index = build_annoy_index([seq_embedding['embedding'] for seq_embedding in seq_embeddings])
        
        tqdm_bar = tqdm(total=num_sents*6, desc="Retrieve")

        embeddings_fp = torch.tensor(np.array([seq_embedding['embedding_float'] for seq_embedding in seq_embeddings]),dtype=torch.float32).cuda()
        embeddings_bi = torch.tensor(np.array([seq_embedding['embedding_bi'] for seq_embedding in seq_embeddings]),dtype=torch.float32).cuda()
        for inputs, _ in batches:
            seqs = [list2str(vocab.idx2word[id] for id in inputs.t()[i][1:]) for i in range(len(inputs.t()))] # (batch_size)
            for noise_level in [0.02, 0.05]:
                args.noise = [0, 0, 0, noise_level]
                inputs_noisy = noisy(vocab, inputs, *args.noise) # (65, batch_size)
                zi_noisy = encode(inputs_noisy) # (batch_size, 32)
                zi_noisy = zi_noisy.detach().cpu().numpy()[:,:32]
                zi_noisy_binary = toBinary(zi_noisy)
                zi_noisy_binary_str = zi_noisy_binary.astype(int)
                # query_noisy_embedding_bi = zi_noisy_binary_str
                # for i in range(len(zi_noisy)):
                    # query_noisy_embeddings.append(query_noisy_embedding[i])
                query_noisy_embeddings_bi = torch.tensor(np.array([zi_noisy_binary[i] for i in range(len(zi_noisy))]),dtype=torch.float32).cuda()
                query_noisy_embeddings_float = torch.tensor(np.array([zi_noisy[i] for i in range(len(zi_noisy))]),dtype=torch.float32).cuda()
                
                for i in range(len(seqs)):
                    
                    seq = seqs[i]
                    query_noisy_embedding_bi = query_noisy_embeddings_bi[i] 
                    query_noisy_embedding_float = query_noisy_embeddings_float[i]
                
                    if not args.use_annoy:
                        distance_matrix_bi = hamming_distance_matrix(query_noisy_embedding_bi, embeddings_bi)
                        distance_matrix_float = l2_distance_matrix(query_noisy_embedding_float, embeddings_fp)
                    torch.sort(distance_matrix_bi)
                    for k in [1, 10, 100]:
                        
                        if args.use_annoy:
                            nearest_neighbors_noisy_indices = find_nearest_neighbors_annoy(query_noisy_embedding_bi, annoy_index, k=k)
                        else:
                            nearest_neighbors_noisy_indices_bi,shared_nearest_neighbors_noisy_indices_bi,shared_ranking_bi = find_nearest_neighbors(distance_matrix_bi, k=k)
                            nearest_neighbors_noisy_indices_float,shared_nearest_neighbors_noisy_indices_float,shared_ranking_float = find_nearest_neighbors(distance_matrix_float, k=k)
                        # print(len(shared_nearest_neighbors_noisy_indices_bi), len(nearest_neighbors_noisy_indices_bi))
                        nearest_neighbors_noisy_bi = seqs_db[nearest_neighbors_noisy_indices_bi]
                        nearest_neighbors_noisy_float = seqs_db[nearest_neighbors_noisy_indices_float]
                        shared_nearest_neighbors_noisy_bi = seqs_db[shared_nearest_neighbors_noisy_indices_bi]
                        shared_nearest_neighbors_noisy_float = seqs_db[shared_nearest_neighbors_noisy_indices_float]
                        seq_match_index = -1
                        if seq in nearest_neighbors_noisy_bi:
                            # check if noise_npercent_topm_correct[f"{noise_level*100}percent_top{k}"] exist
                            if f"{noise_level*100}percent_top{k}" not in noise_npercent_topm_correct_bi:
                                noise_npercent_topm_correct_bi[f"{noise_level*100}percent_top{k}"] = 0
                            noise_npercent_topm_correct_bi[f"{noise_level*100}percent_top{k}"] += 1

                            seq_match_index = nearest_neighbors_noisy_bi.tolist().index(seq)
                        
                        if seq in nearest_neighbors_noisy_float:
                            # check if noise_npercent_topm_correct[f"{noise_level*100}percent_top{k}"] exist
                            if f"{noise_level*100}percent_top{k}" not in noise_npercent_topm_correct_float:
                                noise_npercent_topm_correct_float[f"{noise_level*100}percent_top{k}"] = 0
                            noise_npercent_topm_correct_float[f"{noise_level*100}percent_top{k}"] += 1
                        
                        if seq in shared_nearest_neighbors_noisy_bi:
                            # check if noise_npercent_topm_correct[f"{noise_level*100}percent_top{k}"] exist
                            if f"{noise_level*100}percent_top{k}" not in shared_noise_npercent_topm_correct_bi:
                                shared_noise_npercent_topm_correct_bi[f"{noise_level*100}percent_top{k}"] = 0
                            shared_noise_npercent_topm_correct_bi[f"{noise_level*100}percent_top{k}"] += 1

                            seq_match_index = shared_nearest_neighbors_noisy_bi.tolist().index(seq)
                        
                        if seq in shared_nearest_neighbors_noisy_float:
                            # check if noise_npercent_topm_correct[f"{noise_level*100}percent_top{k}"] exist
                            if f"{noise_level*100}percent_top{k}" not in shared_noise_npercent_topm_correct_float:
                                shared_noise_npercent_topm_correct_float[f"{noise_level*100}percent_top{k}"] = 0
                            shared_noise_npercent_topm_correct_float[f"{noise_level*100}percent_top{k}"] += 1
                        if seq_match_index == -1:
                            if k==10:
                                if f"{noise_level}" not in hitMinusTop1_distances:
                                    hitMinusTop1_distances[f"{noise_level}"] = []
                                hitMinusTop1_distances[f"{noise_level}"].append(-1)

                        if seq_match_index != -1:
                            if k==10:
                                if f"{noise_level}" not in hitMinusTop1_distances:
                                    hitMinusTop1_distances[f"{noise_level}"] = []
                                hitMinusTop1_distances[f"{noise_level}"].append(distance_matrix_bi[shared_nearest_neighbors_noisy_indices_bi][seq_match_index].cpu().numpy()-distance_matrix_bi[shared_nearest_neighbors_noisy_indices_bi][0].cpu().numpy())
                            
                            if f"{noise_level}_top{k}_match" not in match_stat:
                                match_stat[f"{noise_level}_top{k}_match"] = 0
                            match_stat[f"{noise_level}_top{k}_match"] += 1

                            # if f"{noise_level}_top{k}_avg_distance" not in match_stat:
                            #     match_stat[f"{noise_level}_top{k}_avg_distance"] = 0
                            # match_stat[f"{noise_level}_top{k}_avg_distance"] += distance_matrix_bi[shared_nearest_neighbors_noisy_indices_bi][seq_match_index]

                            # if f"{noise_level}_top{k}_avg_index" not in match_stat:
                            #     match_stat[f"{noise_level}_top{k}_avg_index"] = 0
                            # match_stat[f"{noise_level}_top{k}_avg_index"] += seq_match_index

                            dist[f"{noise_level}_top{k}"].append(distance_matrix_bi[shared_nearest_neighbors_noisy_indices_bi][seq_match_index].item())

                            if seq_match_index != len(shared_ranking_bi)-1:
                                if shared_ranking_bi[seq_match_index] == shared_ranking_bi[seq_match_index+1]:
                                    while seq_match_index != len(shared_ranking_bi)-1:
                                        if shared_ranking_bi[seq_match_index] == shared_ranking_bi[seq_match_index+1]:
                                            seq_match_index+=1
                                        else:
                                            break
                            
                            index[f"{noise_level}_top{k}"].append(seq_match_index)
                            # if f"{noise_level}_top{k}_max_distance" not in match_stat:
                            #     match_stat[f"{noise_level}_top{k}_max_distance"] = distance_matrix_bi[shared_nearest_neighbors_noisy_indices_bi][seq_match_index]
                            # elif distance_matrix_bi[shared_nearest_neighbors_noisy_indices_bi][seq_match_index] > match_stat[f"{noise_level}_top{k}_max_distance"]:
                            #     match_stat[f"{noise_level}_top{k}_max_distance"] = distance_matrix_bi[shared_nearest_neighbors_noisy_indices_bi][seq_match_index]
                            
                            # if f"{noise_level}_top{k}_min_distance" not in match_stat:
                            #     match_stat[f"{noise_level}_top{k}_min_distance"] = distance_matrix_bi[shared_nearest_neighbors_noisy_indices_bi][seq_match_index]
                            # elif distance_matrix_bi[shared_nearest_neighbors_noisy_indices_bi][seq_match_index] < match_stat[f"{noise_level}_top{k}_min_distance"]:
                            #     match_stat[f"{noise_level}_top{k}_min_distance"] = distance_matrix_bi[shared_nearest_neighbors_noisy_indices_bi][seq_match_index]
                            
                            # if f"{noise_level}_top{k}_max_index" not in match_stat:
                            #     match_stat[f"{noise_level}_top{k}_max_index"] = seq_match_index
                            # elif seq_match_index > match_stat[f"{noise_level}_top{k}_max_index"]:
                            #     match_stat[f"{noise_level}_top{k}_max_index"] = seq_match_index
                            
                            # if f"{noise_level}_top{k}_min_index" not in match_stat:
                            #     match_stat[f"{noise_level}_top{k}_min_index"] = seq_match_index
                            # elif seq_match_index < match_stat[f"{noise_level}_top{k}_min_index"]:
                            #     match_stat[f"{noise_level}_top{k}_min_index"] = seq_match_index
                        
                        
                        # if k==100:
                        #     # # Report the hamming distance
                        #     if args.retrieval_report:
                        #         report = f"Original:     {seqs[i]}\n"
                        #         report += f"Noisy-{noise_level}:   {''.join([vocab.idx2word[id] for id in inputs_noisy.t()[i][1:]])}\n"
                        #         # report += "\n".join(
                        #         #     f"{j:03d} Sequence: {nearest_neighbors_noisy_bi[j]}, Distance: {distance_matrix_bi[nearest_neighbors_noisy_indices_bi][j]}{' ***' if seq_match_index != -1 and j == seq_match_index else ''}" 
                        #         #     for j in range(100)
                        #         # )
                        #         if seq_match_index != -1:
                        #             # report += "---Hamming distance---\n"
                        #             report += f"{seq_match_index:02d} Sequence:  {nearest_neighbors_noisy_bi[seq_match_index]}, {distance_matrix_bi[nearest_neighbors_noisy_indices_bi][seq_match_index]}\n"
                        #             # report += "\n----------------------\n"
                        #         else:
                        #             report += "No match found\n"
                        #         report += "-----------\n"
                        #         with open(args.retrieval_report+'.txt', 'a') as f:
                        #             f.write(report)

                        tqdm_bar.update(1)
            
        print("(Absolute Ranking) (Hamming) Accuracy of retrieval by the top 1, 10, 100 nearest neighbor for all sentence with 2%, 5% noise")
        for key, value in noise_npercent_topm_correct_bi.items():
            noise_npercent_topm_correct_bi[key] = value / num_sents
        print(noise_npercent_topm_correct_bi)

        print("(Absolute Ranking) (Euclidean) Accuracy of retrieval by the top 1, 10, 100 nearest neighbor for all sentence with 2%, 5% noise")
        for key, value in noise_npercent_topm_correct_float.items():
            noise_npercent_topm_correct_float[key] = value / num_sents
        print(noise_npercent_topm_correct_float)

        print("(Shared Ranking) (Hamming) Accuracy of retrieval by the top 1, 10, 100 nearest neighbor for all sentence with 2%, 5% noise")
        for key, value in shared_noise_npercent_topm_correct_bi.items():
            shared_noise_npercent_topm_correct_bi[key] = value / num_sents
        print(shared_noise_npercent_topm_correct_bi)

        print("(Shared Ranking) (Euclidean) Accuracy of retrieval by the top 1, 10, 100 nearest neighbor for all sentence with 2%, 5% noise")
        for key, value in shared_noise_npercent_topm_correct_float.items():
            shared_noise_npercent_topm_correct_float[key] = value / num_sents
        print(shared_noise_npercent_topm_correct_float)
        
        print("Average, minimum and maximum of the index and distance of the matched sequence for 0.2 noise level")
        for noise_level in [0.02, 0.05]:
            for k in [1, 10, 100]:
                if f"{noise_level}_top{k}_match" in match_stat:
                    
                    print(f"Matched sequence for {noise_level} noise level and top {k} nearest neighbor")
                    print(f"Matched proportion: {match_stat.get(f'{noise_level}_top{k}_match', 0)/num_sents}")
                    
                    if len(dist[f"{noise_level}_top{k}"])>0:
                        
                        print(f'Maximum distance: {max(dist[f"{noise_level}_top{k}"])}')
                        print(f'Minimum distance: {min(dist[f"{noise_level}_top{k}"])}')
                        print(f'Average distance: {average_of_list(dist[f"{noise_level}_top{k}"])}')
                        print(f'Median distance: {np.median(np.array(dist[f"{noise_level}_top{k}"]))}')
                    if len(index[f"{noise_level}_top{k}"])>0:
                        print(f'Maximum index: {max(index[f"{noise_level}_top{k}"])}')
                        print(f'Minimum index: {min(index[f"{noise_level}_top{k}"])}')
                        print(f'Average index: {average_of_list(index[f"{noise_level}_top{k}"])}')
                        print(f'Median index: {np.median(np.array(index[f"{noise_level}_top{k}"]))}')
                    
                    print("-----------")
                
        with open('result/hit_minus_top1_distances.pkl', 'wb') as f:
            pickle.dump(hitMinusTop1_distances, f)
    if args.retrieval_by_radius:
        args.batch_size = 512
        seq_embeddings = pickle.load(open('all_embeddings.pkl', 'rb'))
        seqs_db = np.array([seq_embedding['seq'] for seq_embedding in seq_embeddings])

        sents = load_sent(args.data)
        num_sents = len(sents)
        batches, order = get_batches(sents, vocab, args.batch_size, device)
          
        radius_distribution = {}
        candidates_distribution = {}
        
        tqdm_bar = tqdm(total=num_sents*2, desc="Retrieve")
        embeddings_fp = torch.tensor(np.array([seq_embedding['embedding_float'] for seq_embedding in seq_embeddings]),dtype=torch.float32).cuda()
        embeddings_bi = torch.tensor(np.array([seq_embedding['embedding_bi'] for seq_embedding in seq_embeddings]),dtype=torch.float32).cuda()
        for inputs, _ in batches:
            seqs = [list2str(vocab.idx2word[id] for id in inputs.t()[i][1:]) for i in range(len(inputs.t()))] # (batch_size)
            for noise_level in [0.02, 0.05]:
                args.noise = [0, 0, 0, noise_level]
                inputs_noisy = noisy(vocab, inputs, *args.noise) # (65, batch_size)
                
                zi_anchor = encode(inputs) # (batch_size, 32)
                zi_anchor = zi_anchor.detach().cpu().numpy()[:,:32]
                zi_anchor_binary = toBinary(zi_anchor)
                zi_anchor_binary_str = zi_anchor_binary.astype(int)
                anchor_embeddings_bi = torch.tensor(np.array([zi_anchor_binary_str[i] for i in range(len(zi_anchor))]),dtype=torch.float32).cuda()
                
                anchor_embeddings_float = torch.tensor(np.array([zi_anchor[i] for i in range(len(zi_anchor))]),dtype=torch.float32).cuda()
                
                zi_noisy = encode(inputs_noisy) # (batch_size, 32)
                zi_noisy = zi_noisy.detach().cpu().numpy()
                zi_noisy = zi_noisy[:,:32]
                zi_noisy_binary = toBinary(zi_noisy)
                zi_noisy_binary_str = zi_noisy_binary.astype(int)
                query_noisy_embeddings_bi = torch.tensor(np.array([zi_noisy_binary_str[i] for i in range(len(zi_noisy))]),dtype=torch.float32).cuda()
                query_noisy_embeddings_float = torch.tensor(np.array([zi_noisy[i] for i in range(len(zi_noisy))]),dtype=torch.float32).cuda()
                
                for i in range(len(seqs)):
                    seq = seqs[i]
                    anchor_embedding_bi = anchor_embeddings_bi[i]
                    anchor_emedding_float = anchor_embeddings_float[i]
                    query_noisy_embedding_bi = query_noisy_embeddings_bi[i] 
                    query_noisy_embedding_float = query_noisy_embeddings_float[i]

                    # print("anchor: ", anchor_embedding_bi)
                    # print("query_noisy: ", query_noisy_embedding_bi)
                    hamming_radius = torch.sum(anchor_embedding_bi != query_noisy_embedding_bi)
                    # print("Hamming radius: ", hamming_radius)
                    euc_radius = torch.dist(anchor_emedding_float, query_noisy_embedding_float, p=2)
                    # print("Euclidean radius: ", euc_radius)
                    
                    if f"hamming_{noise_level}" not in radius_distribution:
                        radius_distribution[f"hamming_{noise_level}"] = []
                    radius_distribution[f"hamming_{noise_level}"].append(hamming_radius)
                    
                    if f"euc_{noise_level}" not in radius_distribution:
                        radius_distribution[f"euc_{noise_level}"] = []
                    radius_distribution[f"euc_{noise_level}"].append(euc_radius)
                    
                    distance_matrix_bi = hamming_distance_matrix(query_noisy_embedding_bi, embeddings_bi)
                    distance_matrix_float = l2_distance_matrix(query_noisy_embedding_float, embeddings_fp)

                    hamming_candidates = (distance_matrix_bi <= hamming_radius).sum().cpu().numpy()
                    euc_candidates = (distance_matrix_float <= euc_radius).sum().cpu().numpy()
                    
                    if f"hamming_{noise_level}" not in candidates_distribution:
                        candidates_distribution[f"hamming_{noise_level}"] = []
                    candidates_distribution[f"hamming_{noise_level}"].append(hamming_candidates)
                    
                    if f"euc_{noise_level}" not in candidates_distribution:
                        candidates_distribution[f"euc_{noise_level}"] = []
                    candidates_distribution[f"euc_{noise_level}"].append(euc_candidates)
                    
                    tqdm_bar.update(1)   
        # find the number of candidates in hamming distance 0.02 == 1.0
        print(f'0.02 unique: {candidates_distribution["hamming_0.02"].count(1)}')
        print(f'0.05 unique:{candidates_distribution["hamming_0.05"].count(1)}')
        with open('result/radius_distribution.pkl', 'wb') as f:
            pickle.dump(radius_distribution, f)
        
        with open('result/candidates_distribution.pkl', 'wb') as f:
            pickle.dump(candidates_distribution, f)

    if args.metrics_hm:
        args.noise = [float(x) for x in args.noise.split(',')]
        sents = load_sent(args.data)
        batches, order = get_batches(sents, vocab, args.batch_size, device)
        anchor = []
        generate_sample = []
        for inputs, _ in batches:
            inputs_noisy = noisy(vocab, inputs, *args.noise)
            #print(inputs[:,0])
            #print(inputs_noisy[:,0])
            zi_anchor = encode_hm(inputs)
            zi_sample = encode_hm(inputs_noisy)
            anchor.append(zi_anchor.detach().cpu().numpy())
            generate_sample.append(zi_sample.detach().cpu().numpy())
            break

        anchor = np.concatenate(anchor, axis=0)
        z_ = np.zeros_like(anchor)
        z_[np.array(order)] = anchor
        anchor = z_

        generate_sample = np.concatenate(generate_sample, axis=0)
        z_ = np.zeros_like(generate_sample)
        z_[np.array(order)] = generate_sample
        generate_sample = z_
        
        for i in range(1):
            for j in range(100):
                dist = hamming_distance_bi(anchor[i*100 + j], generate_sample[i*100 + j], train_args.dim_z)
                print(dist, end = ',')
            print()

    if args.triplet:
        sents = load_sent(args.data)
        batches, _ = get_batches(sents, vocab, args.batch_size, device)
        TP,count = evaluate_triplet(model, batches, args.cutoff)
        print('TPR = {:.2f}%'.format(TP/count*100))
        
    if args.interpolate:
        f1, f2 = args.data.split(',')
        s1, s2 = load_sent(f1), load_sent(f2)
        z1, z2 = encode(s1), encode(s2)
        zi = [interpolate(z1_, z2_, args.n) for z1_, z2_ in zip(z1, z2)]
        zi = np.concatenate(zi, axis=0)
        si = decode(zi)
        si = list(zip(*[iter(si)]*(args.n)))
        write_doc(si, os.path.join(args.checkpoint, args.output))

    if args.latent_nn:
        sents = load_sent(args.data)
        z = encode(sents)
        with open(os.path.join(args.checkpoint, args.output), 'w') as f:
            nn = NearestNeighbors(n_neighbors=args.n).fit(z)
            dis, idx = nn.kneighbors(z[:args.m])
            for i in range(len(idx)):
                f.write(' '.join(sents[i]) + '\n')
                for j, d in zip(idx[i], dis[i]):
                    f.write(' '.join(sents[j]) + '\t%.2f\n' % d)
                f.write('\n')
