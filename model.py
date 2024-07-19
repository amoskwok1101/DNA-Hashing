import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from noise import noisy
from torch.cuda.amp import autocast, GradScaler
import math
from global_seq_align import global_alignment_batch_with_gap_penalty
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std  

#if traing for fp, reparameterize is better than reparameterize_corr
def reparameterize_corr(mu, logvar, u_pert, device):
    std = F.softplus(logvar)
    eps = torch.randn_like(mu)
    eps_corr = torch.randn((u_pert.shape[0], u_pert.shape[2], 1)).to(device)
    # Reparameterisation trick for low-rank-plus-diagonal Gaussian
    return mu + eps * std + torch.matmul(u_pert, eps_corr).squeeze()    

def straightThrough(logit, is_logit=True, stochastic=False):
    shape = logit.size()
    if is_logit:
        prob = torch.sigmoid(logit)
    else:
        prob = logit
    if stochastic:
        random = torch.rand_like(prob)
        output_binary = (torch.sign(prob - random) + 1) / 2
    else:
        output_binary = (torch.sign(prob - 0.5) + 1) / 2
    output = torch.autograd.Variable(output_binary - prob.data) + prob
    return output

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    # print(mu.size())
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)


class TextModel(nn.Module):
    """Container module with word embedding and projection layers"""

    def __init__(self, vocab, args, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.embed = nn.Embedding(vocab.size, args.dim_emb)
        self.proj = nn.Linear(args.dim_h, vocab.size)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)
        # self.scalerD = GradScaler()
        # self.scaler = GradScaler()
        self.iter_num = 0
        self.input_seq_len = 0
        self.lambda_quant = 0.0

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super().__init__()
        # self.dropout = nn.Dropout(dropout_p)
        self.dim_model = dim_model
        self.max_len = int(max_len)
    def forward(self, token_embedding):
        # Calculate positional encodings on the fly
        position = torch.arange(self.max_len).unsqueeze(1).to(token_embedding.device)
        div_term = torch.exp(torch.arange(0, self.dim_model, 2).float() * -(math.log(10000.0) / self.dim_model)).to(token_embedding.device)
        
        pe = torch.zeros(self.max_len, 1, self.dim_model).to(token_embedding.device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # pe = pe.unsqueeze(0).transpose(0, 1)
        # Only use the required portion of the positional encoding matrix
        # pos_encoding_cloned = pe[:token_embedding.size(0)].clone()
        token_embedding = token_embedding + pe[:token_embedding.size(0)]
        return token_embedding
    
class DAE(TextModel):
    """Denoising Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.drop = nn.Dropout(args.dropout)
        self.E = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0, bidirectional=True)
        self.G = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0)
        self.scaler = GradScaler()
        self.z2emb = nn.Linear(args.dim_z, args.dim_emb) #for decoder used
        self.init_beta = 1.0
        self.beta = self.init_beta
        if args.use_transformer:
            # self.transformer = TransformerEncoder(args=args, num_tokens=vocab.size)
            self.encoder_layer = nn.TransformerEncoderLayer(
                                d_model=args.dim_emb,       # The dimension of the input embeddings
                                nhead=4,           # The number of heads in the multihead attention models
                                dim_feedforward=args.dim_h, # The dimension of the feedforward network model
                                # dropout=args.dropout        # The dropout value
                            )
            self.T = nn.TransformerEncoder(
                self.encoder_layer,
                num_layers=4     # The number of sub-encoder-layers in the encoder
            )
            self.pos_embedding = PositionalEncoding(args.dim_emb, max_len=5000)
            
            self.h2mu = nn.Linear(args.dim_emb, args.dim_z)
            self.h2logvar = nn.Linear(args.dim_emb, args.dim_z)
            # #new add
            self.h2U = nn.ModuleList([nn.Linear(args.dim_emb, args.dim_z) for i in range(args.rank)])
            self.pos_embedding2.weight.data.uniform_(-0.1, 0.1)
        else:
            self.conv1d = nn.Conv1d(in_channels=args.dim_emb, out_channels=args.dim_emb, kernel_size=3,stride=1)
            self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            self.h2mu = nn.Linear(args.dim_h*2, args.dim_z)
            self.h2logvar = nn.Linear(args.dim_h*2, args.dim_z)
            #new add
            self.h2U = nn.ModuleList([nn.Linear(args.dim_h*2, args.dim_z) for i in range(args.rank)])
        self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.opt,
                       max_lr=0.01,
                       epochs=args.epochs,
                       steps_per_epoch=1,
                       anneal_strategy='cos',
                       div_factor=10,
                       final_div_factor=(0.001/0.00005)) 
        self.scaler = GradScaler()

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.args.dim_h * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
            
        return context, soft_attn_weights.data
    
    def mask_embeddings(self,embeddings, input_ids, pad_token_index=0):
        # Create a mask where `True` values correspond to non-pad tokens
        mask = input_ids != pad_token_index
        # Expand mask to cover the embedding dimensions
        mask = mask.unsqueeze(-1).expand_as(embeddings)
        # Apply mask to embeddings (zero out pad tokens)
        return embeddings * mask.float()
    
    def encode(self, input):
        if self.args.use_transformer:
            input = input.permute(1, 0) 
            padding_mask = (input == self.vocab.pad)     
            seq_length = input.size(1)
            # position_ids = torch.arange(0,seq_length, dtype=torch.long,device = input.device)
            # position_ids = position_ids.unsqueeze(0).expand_as(input)
            # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
            embed = self.embed(input)
            input = self.pos_embedding(embed)  
            input = input.permute(1, 0, 2)
            # mean pooling
            h = self.T(input,src_key_padding_mask=padding_mask).mean(dim=0)         
        else:
            # use pack-padded LSTM

            def mask_embeddings(embeddings, input_ids, pad_token_index=0):
                # Create a mask where `True` values correspond to non-pad tokens
                mask = input_ids != pad_token_index
                # Expand mask to cover the embedding dimensions
                mask = mask.unsqueeze(-1).expand_as(embeddings)
                # Apply mask to embeddings (zero out pad tokens)
                return embeddings * mask.float()
            
            pad_token_index = self.vocab.pad  # Define the padding token index, typically 0
            input_lengths = (input != pad_token_index).sum(dim=0)  # Compute the actual lengths of sequences

           
            if self.args.use_cnn:
                input_embed = self.embed(input)
                input = mask_embeddings(input_embed, input,self.vocab.pad)   
                input = input.permute(1, 2, 0)
                input = self.conv1d(input)
                input = torch.relu(input)
                input = self.max_pool(input)
                input = input.permute(2, 0, 1)
                output, (final_hidden_state, final_cell_state) = self.E(input)  
            else:
                
                pad_token_index = self.vocab.pad  # Define the padding token index, typically 0
                input_lengths = (input != pad_token_index).sum(dim=0)
                # Apply embedding
                input = self.embed(input)

                # Pack the sequence
                packed_input = pack_padded_sequence(input, input_lengths.cpu(), batch_first=False,enforce_sorted=False)
                
                
                output, (final_hidden_state, final_cell_state) = self.E(packed_input)
                output, _ = pad_packed_sequence(output, batch_first=False)
            
            if self.args.no_Attention:
                h = torch.cat([final_hidden_state[-2], final_hidden_state[-1]], 1)
            else:
                output = output.permute(1, 0, 2)
                final_hidden_state = final_hidden_state[-2:]
                h, attention = self.attention_net(output, final_hidden_state) #attention : [batch_size, n_step]

        rs = []
        for i in range(self.args.rank):
            rs.append((1 / self.args.rank) * torch.tanh(self.h2U[i](h)))
        u_pert = tuple(rs)
        u_perturbation = torch.stack(u_pert, 2)

        return self.h2mu(h), self.h2logvar(h), u_perturbation

    def encode2binary(self, input, is_test=False):
        mu, logvar, u_pert = self.encode(input)
        
        s = reparameterize_corr(mu, logvar, u_pert, self.args.device)
        if is_test == False:
            z = straightThrough(s, is_logit=True, stochastic=True)
        else:
            z = straightThrough(mu, is_logit=True, stochastic=False)
        return mu, logvar, z
    
    def encode2z(self, input, is_test=False, is_reparameterize=False):
        mu, logvar, u_pert = self.encode(input)
        if self.args.model_type == 'vae' or is_reparameterize == True:
            s = reparameterize(mu, logvar)
            if is_test == False:
                z = s
            else: 
                z = mu
        else: 
            z = mu
        
        
        # standardize the z
        # if (self.args.lambda_quant==0) and (not self.args.model_type == 'aae'):
        #     z = (z - z.mean(dim=1,keepdim=True)) / z.std(dim=1,keepdim=True)
        return mu, logvar, z

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.G(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def generate(self, z, max_len, alg):
        assert alg in ['greedy' , 'sample' , 'top5']
        sents = []
        input = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(self.vocab.go)
        hidden = None
        for l in range(max_len):
            sents.append(input)
            logits, hidden = self.decode(z, input, hidden)
            if alg == 'greedy':
                input = logits.argmax(dim=-1)
            elif alg == 'sample':
                input = torch.multinomial(logits.squeeze(dim=0).exp(), num_samples=1).t()
            elif alg == 'top5':
                not_top5_indices=logits.topk(logits.shape[-1]-5,dim=2,largest=False).indices
                logits_exp=logits.exp()
                logits_exp[:,:,not_top5_indices]=0.
                input = torch.multinomial(logits_exp.squeeze(dim=0), num_samples=1).t()
        return torch.cat(sents)

    def forward(self, inputs, is_test=False,is_evaluate=False):
        self.input_seq_len = inputs.size(0) - 1
        self.iter_num = self.args.epoch_i * self.args.train_batch_num + self.args.batch_i
        if self.args.is_triplet or self.args.is_ladder or self.args.is_clr:
            
            if self.args.distance_type == 'hamming':
                mu, logvar, anchor =  self.encode2binary(inputs, is_test)
            else:
                mu, logvar, anchor =  self.encode2z(inputs, is_test)
            return self.autoenc(inputs, mu, logvar, anchor, None, is_test, is_evaluate),anchor.clone()
            
        else:
            _inputs = noisy(self.vocab, input, *self.args.noise) if is_test == False else inputs
            mu, logvar = self.encode(_inputs)
            z = reparameterize(mu, logvar)

            logits, _ = self.decode(z, input)
            
            return self.autoenc(inputs, mu, logvar, z, logits, is_test, is_evaluate)

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def loss(self, losses):
        if self.args.fixed_lambda_quant:
            self.lambda_quant = self.args.lambda_quant
        else:
            if self.iter_num%10000==0:
                self.lambda_quant = self.args.lambda_quant*0.625 + 0.375*self.args.lambda_quant * self.iter_num / (self.args.epochs *self.args.train_batch_num)
        
        if self.args.is_ladder:
            if self.args.ladder_pearson:
                return losses['ladder'] + self.lambda_quant * losses['quant'] + losses['pearson loss']
            return losses['ladder'] + self.lambda_quant * losses['quant']
        elif self.args.is_triplet:
            return losses['triplet'] + self.lambda_quant * losses['quant']
        else:
            return losses['rec']
        
    
        
    def generate_triplet(self, inputs, similar_noise, divergent_noise, is_test=False):
        
        if self.args.is_ladder:
            probabilities = [1/4, 1/4, 1/4, 1/4]
            mutation_type = np.random.choice([0, 1, 2, 3], p=probabilities)
            # mutation_type = np.random.randint(0,4)
            divergent_noises = [0,0,0,0]
            
            divergent_noises[mutation_type] = divergent_noise
            # print(inputs.device)
            divergent_inputs = noisy(self.vocab, inputs, *divergent_noises)
            # print(inputs.size())
            actual_noise = 1 - (global_alignment_batch_with_gap_penalty(divergent_inputs[1:].permute(1,0), inputs[1:].permute(1,0),self.args.device)/ (inputs.size(0) -1))
            if self.args.distance_type == 'hamming':
                _, _, negative = self.encode2binary(divergent_inputs, is_test)
            else:
                _, _, negative = self.encode2z(divergent_inputs, is_test)
            return negative, actual_noise

           
        elif self.args.is_triplet:
            if self.args.epochs-1-self.args.epoch_i - 5 > 0:
                divergent_noise = divergent_noise - ((divergent_noise - divergent_noise*0.25)*self.args.epoch_i/(self.args.epochs - 5 -1))
            else:
                divergent_noise = divergent_noise * 0.25
            if is_test == False:
                mutation_rate = divergent_noise
            else:
                mutation_rate = np.random.uniform(self.args.similar_noise, self.args.divergent_noise)
            # mutation_rate = np.random.uniform(self.args.similar_noise, self.args.divergent_noise)

            # used_margin = (-0.41*(mutation_rate**2) + 0.71*mutation_rate) - (-0.41*(similar_noise**2) + 0.71*similar_noise)
            # if self.args.is_triplet:
            #     probabilities = [1/5, 2/5, 1/5, 1/5]
            # else:
            #     probabilities = [1/4, 1/4, 2/4, 0]
            # probabilities = [2/5, 1/5, 1/5, 1/5]
            # mutation_type = np.random.choice([0, 1, 2, 3], p=probabilities)
            mutation_type = np.random.randint(4)
            def mutation_score(mutation_type, x):
                if mutation_type == 0:
                    return 0.23 * x ** 2 + 0.97 * x
                elif mutation_type == 1:
                    return -0.25 * x ** 2 + x
                elif mutation_type == 2:
                    return -0.46 * x ** 2 + 0.99 * x
                else:
                    return -0.41 * x ** 2 + 0.77 * x 

            
            similar_noises = [0,0,0,0]
            divergent_noises = [0,0,0,0]
            similar_noises[mutation_type] = similar_noise
            divergent_noises[mutation_type] = mutation_rate
            similar_inputs = noisy(self.vocab, inputs, *similar_noises)
            divergent_inputs = noisy(self.vocab, inputs, *divergent_noises)
            
            
            temp_similar = 1 - (global_alignment_batch_with_gap_penalty(similar_inputs[1:].permute(1,0), inputs[1:].permute(1,0),self.args.device)/ inputs.size(0))
            temp_div = 1 - (global_alignment_batch_with_gap_penalty(divergent_inputs[1:].permute(1,0), inputs[1:].permute(1,0),self.args.device)/ inputs.size(0))

            used_margin = torch.abs(temp_similar - temp_div)
            mask = temp_similar > temp_div
            
            # Use the mask to select columns to swap
            temp_similar_inputs = similar_inputs.clone()
            similar_inputs[:, mask] = divergent_inputs[:, mask]
            divergent_inputs[:, mask] = temp_similar_inputs[:, mask]
               
            
            if self.args.distance_type == 'hamming':
                _, _, positive = self.encode2binary(similar_inputs, is_test)
                _, _, negative = self.encode2binary(divergent_inputs, is_test)
            else:
                _, _, positive = self.encode2z(similar_inputs, is_test)
                _, _, negative = self.encode2z(divergent_inputs, is_test)
            return positive, negative, used_margin, mutation_rate
    
    def autoenc(self, inputs, mu, logvar, anchor, logits, is_test=False, is_evaluate=False):
        if self.args.is_ladder:    
            noise_group = [0.03, 0.06, 0.1, 0.2]
            noises = []
            divergences=[]
            
            for noise in noise_group:
                negative, actual_divergence = self.generate_triplet(inputs, noise, noise, is_test)
                noises.append(negative)
                divergences.append(actual_divergence)
            
            
            ladder_losses = []
            ladder_loss_sum = 0.0
            for i in range(len(noises)-1):
                current_ladder_loss = 0.0
                for j in range(i+1, len(noises)):
                    used_margin = torch.abs(divergences[i] - divergences[j])
                    mask = divergences[i] > divergences[j]
                    temp_similar_inputs = noises[i].clone()
                    noises[i][mask,:] = noises[j][mask,:]
                    noises[j][mask,:] = temp_similar_inputs[mask,:]

                    temp_similar_inputs = divergences[i].clone()
                    divergences[i][mask] = divergences[j][mask]
                    divergences[j][mask] = temp_similar_inputs[mask]
                    loss = self.triplet_loss(anchor, noises[i], noises[j], used_margin)
                    current_ladder_loss += loss
                ladder_losses.append(current_ladder_loss)
                ladder_loss_sum += current_ladder_loss
            ladder_losses = torch.tensor(ladder_losses,device=self.args.device)

            if self.args.ladder_beta_type == 'ratio':
                betas = torch.tensor([loss / ladder_loss_sum for loss in ladder_losses],device=self.args.device)
                ladder_loss = 0.0
                for i in range(len(betas)):
                    ladder_loss += betas[i] * ladder_losses[i] 
            elif self.args.ladder_beta_type == 'uniform':
                ladder_loss = ladder_loss_sum

            distances = []
            for i in range(len(noises)):
                if self.args.distance_type == 'euclidean':
                    distances.append(self.euclidean_distance(anchor, noises[i]).unsqueeze(0))
                elif self.args.distance_type == 'cosine':
                    distances.append((1 - F.cosine_similarity(anchor, noises[i])).unsqueeze(0))
            
            x = torch.transpose(torch.cat([divergence.unsqueeze(0) for divergence in divergences],dim=0),0,1)
            y = torch.transpose(torch.cat(distances, dim=0),0,1)
            # print(x)
            pearson_coeff = self.pearson_correlation_batch(x,y).mean()
                    # betas = torch.tensor([1.0 for _ in ladder_losses],device=self.args.device)
                
                # if self.iter_num%50==0:
                #     # print(ladder_loss_sum == torch.sum(ladder_losses))
                #     # # print(f'betas: {betas}')
                #     # print(f'quant: {self.quantization_loss(anchor)}')
                #     print(f'anchor: {anchor[0]}')
            return {'ladder': ladder_loss,
                    'pearson coefficient': pearson_coeff,
                    'pearson loss': -pearson_coeff+1,
                    'quant': self.quantization_loss(anchor) if not self.args.is_quant_reparam else self.quantization_loss(self.encode2z(inputs, is_test, is_reparameterize=True)[2])}

            
        if self.args.is_triplet:
            positive, negative, used_margin, mutation_rate = self.generate_triplet(inputs, self.args.similar_noise, self.args.divergent_noise, is_test)
            if self.args.is_matry:
                # matryoshka representation learning
                # paper link: https://arxiv.org/abs/2205.13147
                triplet_losses = 0.0
                dim_num = 3
                for i in range(dim_num):
                    triplet_losses+=self.triplet_loss(anchor[:,:self.args.dim_z//(2**i)], positive[:,:self.args.dim_z//(2**i)], negative[:,:self.args.dim_z//(2**i)], used_margin)
                
                return {'triplet': triplet_losses/dim_num if is_evaluate == False else self.triplet_loss_detail(anchor, positive, negative, used_margin),
                        'quant': self.quantization_loss(anchor) if not self.args.is_quant_reparam else self.quantization_loss(self.encode2z(inputs, is_test, is_reparameterize=True)[2])}
            else:
                return {'triplet': self.triplet_loss(anchor, positive, negative, used_margin),
                        'quant': self.quantization_loss(anchor) if not self.args.is_quant_reparam else self.quantization_loss(self.encode2z(inputs, is_test, is_reparameterize=True)[2])}
        else:
            # _, _, _, logits = self(inputs, is_test)
            return {'rec': self.loss_rec(logits, targets).mean()}

    def step(self, accelerator, losses):
        self.opt.zero_grad()
        if self.args.use_amp:
            # use automatix mixed precision
            accelerator.backward(self.scaler.scale(losses['loss']))
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            accelerator.backward(losses['loss'])
            self.opt.step()
        
    def nll_is(self, inputs, targets, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar = self.encode(inputs)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inputs)
            v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
                self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is

    def euclidean_distance(self, x, y):
        return torch.sqrt(torch.sum(torch.pow(x - y, 2), dim=1)) / self.args.dim_z
    
    def hamming_distance(self, x, y):
        return torch.sum(torch.abs(x - y), dim=1) / self.args.dim_z
    
    def quantization_loss(self, anchor):
        if self.args.copy_quant:
            copy_from_indices = torch.randint(0, anchor.size(1), (anchor.size(0),))
            copy_to_indices = torch.randint(0, anchor.size(1), (anchor.size(0),))
            anchor[torch.arange(anchor.size(0)), copy_to_indices] = anchor[torch.arange(anchor.size(0)), copy_from_indices]
        quant_anchor = torch.norm(anchor - torch.sign(anchor), p=2,dim = 1) / self.args.dim_z
        
        
        return quant_anchor.mean()

    def triplet_loss(self, anchor, positive, negative, margin):           
        ###################### the triplet loss ###################################
        if self.args.rescaled_margin_type =='quadratic':
            margin = margin ** 0.75
        elif self.args.rescaled_margin_type =='scaled to dim_z':
            # difference in 1 mutation -> 1/32 hamming distance
            margin = margin * self.input_seq_len / self.args.dim_z
        lambda_margin = self.args.lambda_margin
        self.iter_num = self.args.epoch_i * self.args.train_batch_num + self.args.batch_i
        if self.iter_num%1000==0:
            self.beta = self.init_beta*(math.pow((1.+0.001*self.iter_num),0.5))
        # self.beta = 1.0 + (10.0- 1.0) * self.args.epoch_i / (self.args.epochs - 1)
        # quant loss: loss of conversion from float z to binary z
        
        # quant_loss = self.euclidean_distance(anchor, torch.sign(anchor)).mean()
        if self.args.is_tanh:
            positive = torch.tanh(positive)
            negative = torch.tanh(negative)
            anchor = torch.tanh(anchor)
        if self.args.distance_type == 'hamming':
            pos_dist = self.hamming_distance(anchor, positive)
            neg_dist = self.hamming_distance(anchor, negative)
            tripletLoss = torch.clamp(margin * lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12)
        elif self.args.distance_type == "euclidean":
            pos_dist = self.euclidean_distance(anchor, positive)
            neg_dist = self.euclidean_distance(anchor, negative)
            tripletLoss = torch.clamp(margin * lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12)
        elif self.args.distance_type == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
            tripletLoss = torch.clamp(margin * lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12)
        elif self.args.distance_type == 'cosine_sim':
            # impose higher penalty on hard triplets
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative) 
           
            hard_mask = (pos_dist > neg_dist) 
            pos_dist_easy = pos_dist[~hard_mask]
            neg_dist_easy = neg_dist[~hard_mask]
            margin_easy = margin[~hard_mask]
            pos_dist_hard = pos_dist[hard_mask]
            neg_dist_hard = neg_dist[hard_mask]
            margin_hard = margin[hard_mask]
            tripletLoss_easy = torch.clamp(margin_easy * lambda_margin + pos_dist_easy - neg_dist_easy + self.args.lambda_sim * pos_dist_easy, min=1e-12)
            tripletLoss_hard = torch.clamp(margin_hard * lambda_margin + pos_dist_hard - neg_dist_hard + self.args.lambda_sim * pos_dist_hard, min=1e-12)
            tripletLoss = (tripletLoss_easy + 2*tripletLoss_hard) / len(pos_dist)
        else:
            # impose higher penalty on hard triplets and when divergence margin > 0
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative) 
            hard_mask = (pos_dist > neg_dist) 
            # margin_threshold = 1.0 / self.input_seq_len
            pos_dist_easy = pos_dist[~hard_mask & (margin > 0 )]
            neg_dist_easy = neg_dist[~hard_mask& (margin > 0)]
            margin_easy = margin[~hard_mask& (margin > 0)]
            pos_dist_hard = pos_dist[hard_mask& (margin > 0)]
            neg_dist_hard = neg_dist[hard_mask& (margin > 0)]
            margin_hard = margin[hard_mask& (margin > 0)]
            tripletLoss_easy = torch.clamp(margin_easy * lambda_margin + pos_dist_easy - neg_dist_easy + self.args.lambda_sim * pos_dist_easy, min=1e-12)
            tripletLoss_hard = torch.clamp(margin_hard * lambda_margin + pos_dist_hard - neg_dist_hard + self.args.lambda_sim * pos_dist_hard, min=1e-12)
            # minimize distance spread when divergence margin = 0
            pos_dist_eq = pos_dist[margin == 0]
            neg_dist_eq = neg_dist[margin == 0]
            eq_loss = torch.clamp(torch.abs(pos_dist_eq - neg_dist_eq), min=1e-12)
            tripletLoss = (tripletLoss_easy + 3*tripletLoss_hard + eq_loss) / len(pos_dist)
            #tripletLoss = self.cosine_margin_triplet_loss(anchor, positive, negative, margin)
        # if self.args.is_quant:
            # if self.iter_num%1000==0:
            #     print(f'z: {anchor[0]}')
        if self.args.loss_reduction == 'mean':
            tripletLoss = tripletLoss.mean()
        elif self.args.loss_reduction == 'sum':
            tripletLoss = tripletLoss.sum()
        elif self.args.loss_reduction == 'max':
            tripletLoss = tripletLoss.max()
        return tripletLoss
    
    def triplet_loss_detail(self, anchor, positive, negative, margin):           
        ###################### the triplet loss ###################################

        lambda_margin = self.args.lambda_margin
        if self.args.distance_type == 'hamming':
            pos_dist = self.hamming_distance(anchor, positive)
            neg_dist = self.hamming_distance(anchor, negative)
            tripletLoss = torch.clamp(margin * lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12)
        elif self.args.distance_type == "euclidean":
            pos_dist = self.euclidean_distance(anchor, positive)
            neg_dist = self.euclidean_distance(anchor, negative)
            tripletLoss = torch.clamp(margin * lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12)
        else:
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
            tripletLoss = torch.clamp(margin * lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12)
        return tripletLoss

    def pearson_correlation_batch(self,x, y):
        """
        Calculate the Pearson correlation coefficient between two batches of PyTorch tensors.

        Parameters:
        x (torch.Tensor): First input tensor of shape (batch, length).
        y (torch.Tensor): Second input tensor of shape (batch, length).

        Returns:
        torch.Tensor: Tensor of Pearson correlation coefficients for each batch.
        """
        # Ensure the tensors have the same shape
        if x.size() != y.size():
            raise ValueError("Input tensors must have the same shape")

        # Compute means
        mean_x = torch.mean(x, dim=1, keepdim=True)
        mean_y = torch.mean(y, dim=1, keepdim=True)

        # Compute the numerator and denominator of the Pearson correlation coefficient
        numerator = torch.sum((x - mean_x) * (y - mean_y), dim=1)
        denominator = torch.sqrt(torch.sum((x - mean_x) ** 2, dim=1) * torch.sum((y - mean_y) ** 2, dim=1))

        # Handle the case of zero variance
        denominator = torch.where(denominator == 0, torch.tensor(float('inf'),device=self.args.device), denominator)

        # Calculate the Pearson correlation coefficient
        correlation = numerator / denominator

        return correlation
    
class VAE(DAE):
    """Variational Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)

    def loss(self, losses):
        if self.args.is_triplet:
            # increase lambda_kl linearly from 0 to 0.005 in 500 iterations
            lambda_kl = 0.001 * min(1.0, ( self.iter_num % 4000) / 2000)
            return (1 - lambda_kl)*losses['triplet'] +  lambda_kl *losses['kl']
        else:
            return losses['rec'] + self.args.lambda_kl * losses['kl']

    def autoenc(self, inputs, mu, logvar, anchor, logits, is_test=False, is_evaluate=False):
        if self.args.is_triplet:
            positive, negative, used_margin, mutation_rate = self.generate_triplet(inputs, self.args.similar_noise, self.args.divergent_noise, is_test)
            return {'triplet': self.triplet_loss(anchor, positive, negative, used_margin) if is_evaluate == False else self.triplet_loss_detail(anchor, positive, negative, used_margin),
                    'kl': loss_kl(mu, logvar)}
        else:
            # mu, logvar, _, logits = self(inputs, is_test)
            return {'rec': self.loss_rec(logits, targets).mean(),
                    'kl': loss_kl(mu, logvar)}
class AAE(DAE):
    """Adversarial Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.ReLU(),
            nn.Linear(args.dim_d, 1))
        self.optD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.schedulerD = optim.lr_scheduler.OneCycleLR(self.optD,
                       max_lr=0.01,
                       epochs=args.epochs,
                       steps_per_epoch=1,
                       anneal_strategy='cos',
                       div_factor=10,
                       final_div_factor=(0.001/0.00005))
        self.scalerD = GradScaler()

    def loss_adv(self, z):
        # Concatenate 2 normal distributions to form bimodal distribution
        samples1 = torch.normal(mean=1, std=0.25, size=(int(z.size(0)), int(z.size(1)//2)), device=self.args.device)
        samples2 = torch.normal(mean=-1, std=0.25, size=(int(z.size(0)), int(z.size(1)//2)), device=self.args.device)
        zn = torch.cat([samples1, samples2], dim=-1)

        # Generate a random permutation for each row
        perm = torch.argsort(torch.rand(zn.size(), device=self.args.device), dim=1)

        # Gather the shuffled rows
        zn = torch.gather(zn, dim=1, index=perm)
        zeros = torch.zeros(len(z), 1, device=z.device)
        ones = torch.ones(len(z), 1, device=z.device)
        loss_d = F.binary_cross_entropy_with_logits(self.D(z.detach()), zeros) + \
            F.binary_cross_entropy_with_logits(self.D(zn), ones)
        loss_g = F.binary_cross_entropy_with_logits(self.D(z), ones)
        return 0.5*loss_d, loss_g

    def loss(self, losses):
        if self.args.is_ladder:
            return losses['ladder'] + self.args.lambda_adv * losses['adv'] + \
                self.args.lambda_p * losses['|lvar|']
        if self.args.is_triplet:
            return 0.995* losses['triplet'] + (1.0-0.995) * losses['adv']
            return losses['triplet'] + self.args.lambda_adv * losses['adv'] + \
                self.args.lambda_p * losses['|lvar|']
        else:
            return losses['rec'] + self.args.lambda_adv * losses['adv'] + \
                self.args.lambda_p * losses['|lvar|']
                
    def forward(self, inputs, is_test=False,is_evaluate=False,is_D=False):
        
        if self.args.is_triplet or self.args.is_ladder or self.args.is_clr:
            if is_D==False:
                
                if self.args.distance_type == 'hamming' or self.args.distance_type == 'cosine_binary':
                    mu, logvar, anchor =  self.encode2binary(inputs, is_test)
                else:
                    mu, logvar, anchor =  self.encode2z(inputs, is_test)
                return self.autoenc(inputs, mu, logvar, anchor, None, is_test, is_evaluate),anchor.clone()
            else:
                loss_d, adv = self.loss_adv(inputs)
                return loss_d
            
        else:
            _inputs = noisy(self.vocab, input, *self.args.noise) if is_test == False else inputs
            mu, logvar = self.encode(_inputs)
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, input)
            
            return self.autoenc(inputs, mu, logvar, z, logits, is_test, is_evaluate)
        
    def autoenc(self, inputs, mu, logvar, anchor, logits, is_test=False, is_evaluate=False):
        if self.args.is_ladder:
            noise_group = [0.03, 0.06, 0.1, 0.2]
            noises = []
            divergences=[]
            ladder_loss = 0.0
            for noise in noise_group:
                negative, actual_divergence = self.generate_triplet(inputs, noise, noise, is_test)
                noises.append(negative)
                divergences.append(actual_divergence)
            for i in range(len(noises)-1):
                current_ladder_loss = 0.0
                for j in range(i+1, len(noises)):
                    used_margin = torch.abs(divergences[i] - divergences[j])
                    mask = divergences[i] > divergences[j]
                    temp_similar_inputs = noises[i].clone()
                    noises[i][mask,:] = noises[j][mask,:]
                    noises[j][mask,:] = temp_similar_inputs[mask,:]
                    loss = self.triplet_loss(anchor, noises[i], noises[j], used_margin)
                    current_ladder_loss += loss
                
                ladder_loss += current_ladder_loss
            loss_d, adv = self.loss_adv(anchor)
            return {'ladder': ladder_loss,
                    'adv': adv,
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d
                }
        if self.args.is_triplet:
            positive, negative, used_margin, mutation_rate = self.generate_triplet(inputs,self.args.similar_noise, self.args.divergent_noise, is_test)
            # adv_z = anchor if is_test == False else mu
            loss_d, adv = self.loss_adv(anchor)
            
            return {'triplet': self.triplet_loss(anchor, positive, negative, used_margin) if is_evaluate == False else self.triplet_loss_detail(anchor, positive, negative, used_margin),
                'adv': adv,
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d
                }
        else:
            # _, logvar, z, logits = self(inputs, is_test)
            loss_d, adv = self.loss_adv(z)
            return {'rec': self.loss_rec(logits, targets).mean(),
                'adv': adv,
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d}

    def step(self, accelerator, losses,is_D=False):
        if is_D==False:
            super().step(accelerator,losses)
        else:

            self.optD.zero_grad()
            
            if self.args.use_amp:
                accelerator.backward(self.scalerD.scale(losses['loss_d']))
                self.scalerD.step(self.optD)
                self.scalerD.update()
            else:
                accelerator.backward(losses['loss_d'])
                self.optD.step()