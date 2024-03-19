import argparse
import os
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from vocab import Vocab
from model import *
from utils import *
from batchify import get_batches
from train import evaluate
from noise import noisy_dna

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

parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')

def get_model(path):
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[train_args.model_type](
        vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model, train_args

def encode(inputs):
    assert args.enc == 'mu' or args.enc == 'z'
    if args.enc == 'mu':
        _, _, z = model.encode2z(inputs, is_test=True)
    else:
        _, _, z = model.encode2z(inputs)
    return z

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
    return np.sqrt(np.sum(np.power(x - y, 2), axis = 0)) / dim_z

def cosine_distance(x, y):
    cos_sim = x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return 1 - cos_sim

def hamming_distance(x, y, dim_z):
    x = toBinary(x)
    y = toBinary(y)
    return np.sum(np.abs(x - y), axis = 0) / dim_z

def hamming_distance_bi(x, y, dim_z):
    x = toBinary(x)
    return np.sum(np.abs(x - y), axis = 0) / dim_z

def sigmoid(x):
    #return 1/(1 + np.exp(-x))
    l = len(x)
    y = []
    for i in range(l):
        if x[i] >= 0:
            y.append(1.0/(1+np.exp(-x[i])))
        else :
            y.append(np.exp(x[i])/(np.exp(x[i])+1))
    return np.array(y)

def toBinary(code):
    code = sigmoid(code)
    code = code - 0.5
    code = (np.sign(code) + 1) / 2
    return code

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
    vocab = Vocab(os.path.join(args.checkpoint, 'vocab.txt'))
    set_seed()
    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
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
    
    if args.metrics:
        args.noise = [float(x) for x in args.noise.split(',')]
        sents = load_sent(args.data)
        batches, order = get_batches(sents, vocab, args.batch_size, device)
        anchor = []
        generate_sample = []
        for inputs, _ in batches:
            inputs_noisy = noisy(vocab, inputs, *args.noise)
            #print(inputs[:,0])
            #print(inputs_noisy[:,0])
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

        for i in range(100):
            for j in range(100):
                if args.distance_type == "hamming":
                    dist = hamming_distance(anchor[i*100 + j], generate_sample[i*100 + j], train_args.dim_z)
                elif args.distance_type == "cosine":
                    dist = cosine_distance(anchor[i*100 + j], generate_sample[i*100 + j]);
                elif args.distance_type == "euclidean":
                    dist = euclidean_distance(anchor[i*100 + j], generate_sample[i*100 + j], train_args.dim_z)
                print(dist, end = ',')
            print()

    if args.metrics_hm:
        args.noise = [float(x) for x in args.noise.split(',')]
        sents = load_sent(args.data)
        batches, order = get_batches(sents, vocab, args.batch_size, device)
        anchor = []
        generate_sample = []
        for inputs, _ in batches:
            inputs_noisy = noisy(vocab, inputs, *args.noise)
            zi_anchor = encode_hm(inputs)
            zi_sample = encode_hm(inputs_noisy)
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
