import argparse
import time
import os
import random
import collections
import numpy as np
import torch
import math
from tqdm import tqdm
from model import DAE, VAE, AAE
from vocab import Vocab
from meter import AverageMeter
from utils import set_seed, logging, load_sent, linear_decay_scheduler
from batchify import get_batches,get_batch_dna
from noise import noisy
from Seq_dataset import SeqDataset
from torch.utils.data import DataLoader
import wandb
from accelerate import Accelerator
import torch.optim as optim
from accelerate import DistributedDataParallelKwargs
from torch.cuda.amp import autocast, GradScaler
from functools import partial

parser = argparse.ArgumentParser()
# Path arguments
parser.add_argument('--train', metavar='FILE', required=True,
                    help='path to training file')
parser.add_argument('--valid', metavar='FILE', required=True,
                    help='path to validation file')
parser.add_argument('--save-dir', default='checkpoints', metavar='DIR',
                    help='directory to save checkpoints and outputs')
parser.add_argument('--load-model', default='', metavar='FILE',
                    help='path to load checkpoint if specified') # deprecated
parser.add_argument('--model-path', default='', metavar='FILE',
                    help='path to load model and log if specified')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from the previous best model')
# Architecture arguments
parser.add_argument('--vocab-size', type=int, default=10000, metavar='N',
                    help='keep N most frequent words in vocabulary')
parser.add_argument('--dim_z', type=int, default=32, metavar='D',
                    help='dimension of latent variable z')
parser.add_argument('--dim_emb', type=int, default=512, metavar='D',
                    help='dimension of word embedding')
parser.add_argument('--dim_h', type=int, default=1024, metavar='D',
                    help='dimension of hidden state per layer')
parser.add_argument('--nlayers', type=int, default=1, metavar='N',
                    help='number of layers in LSTM')
parser.add_argument('--dim_d', type=int, default=256, metavar='D',
                    help='dimension of hidden state in AAE discriminator')
# Model arguments
parser.add_argument('--model_type', default='dae', metavar='M',
                    choices=['dae', 'vae', 'aae', 'paae'],
                    help='which model to learn')
parser.add_argument('--lambda_kl', type=float, default=0, metavar='R',
                    help='weight for kl term in VAE')
parser.add_argument('--lambda_adv', type=float, default=0, metavar='R',
                    help='weight for adversarial loss in AAE')
parser.add_argument('--lambda_p', type=float, default=0, metavar='R',
                    help='weight for L1 penalty on posterior log-variance')
parser.add_argument('--similar-noise', type=float, default='0.05', metavar='P',
                    help='similar noise for word drop prob, add_prob, substitute prob or any_prob')
parser.add_argument('--divergent-noise', type=float, default='0.2', metavar='P',
                    help='divergent noise (maximum) for word drop prob, add_prob, substitute prob or any_prob')
parser.add_argument('--lambda_sim', type=float, default=0, metavar='R',
                    help='weight for dist between anchor and similar relative to triplet loss')
parser.add_argument('--lambda_margin', type=float, default=2, metavar='R',
                    help='weight for mutaion rate to be the margin in triplet loss')
parser.add_argument('--rank', default=1, type=int, metavar='R',
                    help="number of ranks of perturbation for covariance.")
parser.add_argument('--k', default=1, type=int, metavar='R',
                    help="k-mers")
parser.add_argument('--seqlen', default=64, type=int, metavar='R',
                    help="read length")

# Training arguments
parser.add_argument('--dropout', type=float, default=0.5, metavar='DROP',
                    help='dropout probability (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate')
#parser.add_argument('--clip', type=float, default=0.25, metavar='NORM',
#                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size')
# Others
parser.add_argument('--no-cuda', action='store_true',
                    help='disable CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--is-triplet', action='store_true',
                    help='train by triplet loss')
parser.add_argument('--distance_type', default='cosine', metavar='M',
                    choices=['cosine', 'euclidean', 'hamming'],
                    help='which distance or similarity is to used in the triplet loss')
parser.add_argument('--no-Attention', action='store_true',
                    help='indicate to use attention mechanism')
parser.add_argument('--use_transformer', action='store_true', default=False,
                    help='indicate to use transformer as autoencoder')
parser.add_argument('--use-amp', action='store_true', 
                    help='whether to use torch.amp (automatic mixed precision)')
parser.add_argument('--use-wandb', action='store_true',
                    help='whether to use wandb for logging')
parser.add_argument('--resume-wandb-id', default='',
                   help='resume wandb logging to the run with the given id') 
parser.add_argument('--use-scheduler', action='store_true',
                    help='whether to use scheduler for learning rate decay')

def evaluate(accelerator, model, batches):
    model.eval()
    meters = collections.defaultdict(lambda: AverageMeter())
    with torch.no_grad():
        for inputs in batches:
            losses,_ = model(inputs, is_test=True)
            for k, v in losses.items():
                meters[k].update(v.item(), len(inputs))
    accelerator.wait_for_everyone()
    loss = accelerator.unwrap_model(model).loss({k: meter.avg for k, meter in meters.items()})
    meters['loss'].update(loss)
    return meters

def main(args):
    # if args.resume == 0:
    #     resume = False # start a new run 
    # elif args.resume == 1 and args.resume_wandb_id == '':
    #     resume = True # resume from the latest previous run
    # elif args.resume == 1 and args.resume_wandb_id != '':
    #     resume = 'must' # resume from the run with the given id\
    if args.use_wandb:
        resume = 'must'

        wandb_init_kwargs = {
            "id": args.resume_wandb_id,
            "project": "DAAE",
            "config": {
                "model_type": args.model_type,
                # "vocab_size": args.vocab_size,
                "dim_z": args.dim_z,
                # "dim_emb": args.dim_emb,
                # "dim_h": args.dim_h,
                "nlayers": args.nlayers,
                # "dim_d": args.dim_d,
                # "lambda_kl": args.lambda_kl,
                # "lambda_adv": args.lambda_adv,
                # "lambda_p": args.lambda_p,
                # "noise": args.noise,
                # "similar_noise": args.similar_noise,
                # "divergent_noise": args.divergent_noise,
                # "margin": args.margin,
                # "lambda_sim": args.lambda_sim,
                "lambda_margin": args.lambda_margin,
                # "rank": args.rank,
                # "dropout": args.dropout,
                "lr": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                # "is_triplet": args.is_triplet,
                # "distance_type": args.distance_type,
                "no_Attention": args.no_Attention,
                "use_amp": args.use_amp,
            },
            "resume": resume
        }

        wandb.init(**wandb_init_kwargs)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_file = os.path.join(args.save_dir, 'log.txt')
    logging(str(args), log_file)

    # Prepare data
    train_sents = load_sent(args.train)
    logging('# train sents {}, tokens {}'.format(
        len(train_sents), sum(len(s) for s in train_sents)), log_file)
    args.steps_per_epoch = int(math.ceil(len(train_sents) / args.batch_size))
    valid_sents = load_sent(args.valid)
    logging('# valid sents {}, tokens {}'.format(
        len(valid_sents), sum(len(s) for s in valid_sents)), log_file)
    vocab_file = os.path.join(args.save_dir, 'vocab.txt')
    if not os.path.isfile(vocab_file):
        Vocab.build_kmer(args.k, vocab_file, args.vocab_size)
    vocab = Vocab(vocab_file)
    logging('# vocab size {}'.format(vocab.size), log_file)

    set_seed()
    

    cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    args.device = device
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[args.model_type](
        vocab, args).to(device)
    
    get_batch_param = partial(get_batch_dna, vocab=vocab, device=device)
    train_dataset = SeqDataset(train_sents, vocab)
    valid_dataset = SeqDataset(valid_sents, vocab)
    train_dataloader= DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=get_batch_param)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=get_batch_param)

    starting_epoch = 0
    best_val_loss = None
    starting_batch = 0

    if args.model_type == 'aae':
        print('true')
        model, model.opt, model.optD, train_dataloader = accelerator.prepare(model, model.opt, model.optD, train_dataloader)
    else:
        model, model.opt, train_dataloader = accelerator.prepare(model,model.opt,train_dataloader)

    if args.resume == 1:
        logging('load accelerator state from {}'.format(args.save_dir), log_file)
        accelerator.load_state(args.save_dir)
        if (os.path.exists(args.save_dir + '/checkpoint.pt')):
            ckpt = torch.load(args.save_dir + '/checkpoint.pt')
            # model.load_state_dict(ckpt['model'])
            # model.flatten()
            logging('load model params from {}'.format(args.save_dir + '/checkpoint_params.pt'), log_file)
            if 'epoch' in ckpt:
                starting_epoch = ckpt['epoch']
            # if 'indices' in ckpt:
            #     indices = ckpt['indices']
            if 'batch_i' in ckpt:
                starting_batch = ckpt['batch_i']
        elif (os.path.exists(args.save_dir + '/model.pt')):
            ckpt = torch.load(args.model_path + '/model.pt')
            # model.load_state_dict(ckpt['model'])
            # model.flatten()
            logging('load model params from {}'.format(args.model_path + '/model_params.pt'), log_file)
            if 'epoch' in ckpt:
                starting_epoch = ckpt['epoch']
            if 'best_val_loss' in ckpt:
                best_val_loss = ckpt['best_val_loss']

    logging('# model parameters: {}'.format(
        sum(x.data.nelement() for x in model.parameters())), log_file)
    
    # train_batches, _ = get_batches(train_sents, vocab, args.batch_size, device)
    # valid_batches, _ = get_batches(valid_sents, vocab, args.batch_size, device)
    
    
    for epoch in range(starting_epoch, args.epochs):
        start_time = time.time()
        logging('-' * 80, log_file)
        model.train()
        meters = collections.defaultdict(lambda: AverageMeter())
        if args.use_scheduler:
            custom_lr = linear_decay_scheduler(epoch, args.epochs, initial_learning_rate=0.001, final_learning_rate=0.00001)
        else:
            custom_lr = None
        if starting_batch==0:
            active_train_dataloader = train_dataloader
        else:
            active_train_dataloader = accelerator.skip_first_batches(train_dataloader,starting_batch)
            # indices = list(range(len(train_batches)))
            # random.shuffle(indices)
        for i, inputs in tqdm(enumerate(active_train_dataloader), total=len(active_train_dataloader)):
            
            if args.use_amp:
                with autocast():
                    losses,anchor = model(inputs)
                    losses['loss'] = accelerator.unwrap_model(model).loss(losses)
                accelerator.unwrap_model(model).step(accelerator,losses,custom_lr)
                if args.model_type == 'aae':
                # optimizing model.D
                    with autocast():
                        losses['loss_d'] = model(anchor,is_D=True)
                    accelerator.unwrap_model(model).step(accelerator,losses,custom_lr,is_D=True)   
            else:
                with accelerator.autocast():
                    losses,anchor = model(inputs)
                    losses['loss'] = accelerator.unwrap_model(model).loss(losses)
                accelerator.unwrap_model(model).step(accelerator,losses,custom_lr)
                if args.model_type == 'aae':
                # optimizing model.D
                    with accelerator.autocast():
                        losses['loss_d'] = model(anchor,is_D=True)
                    accelerator.unwrap_model(model).step(accelerator,losses,custom_lr,is_D=True)
            
            for k, v in losses.items():
                meters[k].update(v.item())

            if (i + 1) % args.log_interval == 0:
                log_output = '| epoch {:3d} | {:5d}/{:5d} batches |'.format(
                    epoch + 1, i + 1, len(active_train_dataloader))
                for k, meter in meters.items():
                    log_output += ' {} {:.4f},'.format(k, meter.avg)
                    meter.clear()
                logging(log_output, log_file)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                ckpt = {'args': args, 'model':unwrapped_model.state_dict(), 'epoch': epoch, 'batch_i': i+1}
                accelerator.save(ckpt, os.path.join(args.save_dir, 'checkpoint.pt'))
                accelerator.save_state(args.save_dir,False)

        valid_meters = evaluate(accelerator, model, valid_dataloader)
        if args.use_wandb:
            log_dict_temp = {}
            log_dict_temp["epoch"] = epoch+1

            log_dict_temp["train/triplet"] = meters['triplet'].avg
            log_dict_temp["train/adv"] = meters['adv'].avg
            log_dict_temp["train/|lvar|"] = meters['|lvar|'].avg
            log_dict_temp["train/loss_d"] = meters['loss_d'].avg
            log_dict_temp["train/loss"] = meters['loss'].avg

            log_dict_temp["valid/triplet"] = valid_meters['triplet'].avg
            log_dict_temp["valid/adv"] = valid_meters['adv'].avg
            log_dict_temp["valid/|lvar|"] = valid_meters['|lvar|'].avg
            log_dict_temp["valid/loss_d"] = valid_meters['loss_d'].avg
            log_dict_temp["valid/loss"] = valid_meters['loss'].avg

            wandb.log(log_dict_temp)

        logging('-' * 80, log_file)
        log_output = '| end of epoch {:3d} | time {:5.0f}s | valid'.format(
            epoch + 1, time.time() - start_time)
        for k, meter in valid_meters.items():
            log_output += ' {} {:.4f},'.format(k, meter.avg)
        if not best_val_loss or valid_meters['loss'].avg < best_val_loss:
            log_output += ' | saving model'
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            ckpt = {'args': args, 'model':unwrapped_model.state_dict(), 'epoch': epoch+1, 'best_val_loss': valid_meters['loss'].avg}
            accelerator.save(ckpt, os.path.join(args.save_dir, 'model.pt'))
            accelerator.save_state(args.save_dir,False)
            best_val_loss = valid_meters['loss'].avg
        logging(log_output, log_file)

        starting_batch = 0
        # remove checkpoint.pt after one epoch
        if os.path.exists(os.path.join(args.save_dir, 'checkpoint.pt')):
            os.remove(os.path.join(args.save_dir, 'checkpoint.pt'))
    
    if args.use_wandb:
        wandb.finish()
    logging('Done training', log_file)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)