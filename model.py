import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from noise import noisy, noisy_dna
from batchify import tokenization, convert_sequences_to_kmers
import math
from torch.cuda.amp import autocast, GradScaler

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)  

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
        output_binary = prob.data.new(*shape).zero_().add(((prob - random) > 0.).float())
    else:
        output_binary = prob.data.new(*shape).zero_().add((prob > 0.5).float())
    output = torch.autograd.Variable(output_binary - prob.data) + prob
    return output

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

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

class DAE(TextModel):
    """Denoising Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.drop = nn.Dropout(args.dropout)
        self.E = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0, bidirectional=True)
        self.G = nn.LSTM(args.dim_emb, args.dim_h, args.nlayers,
            dropout=args.dropout if args.nlayers > 1 else 0)
        

        self.z2emb = nn.Linear(args.dim_z, args.dim_emb) #for decoder used
        if args.use_transformer:
            self.transformer = TransformerEncoder(num_tokens=vocab.size, dim_model=args.dim_h, num_heads=4, num_encoder_layers=2, dropout_p=args.dropout,abs_position_encoding=args.abs_position_encoding)
            self.h2mu = nn.Linear(args.dim_h, args.dim_z)
            self.h2logvar = nn.Linear(args.dim_h, args.dim_z)
            # #new add
            self.h2U = nn.ModuleList([nn.Linear(args.dim_h, args.dim_z) for i in range(args.rank)])
        else:
            self.h2mu = nn.Linear(args.dim_h*2, args.dim_z)
            self.h2logvar = nn.Linear(args.dim_h*2, args.dim_z)
            #new add
            self.h2U = nn.ModuleList([nn.Linear(args.dim_h*2, args.dim_z) for i in range(args.rank)])
        self.opt = optim.Adam(self.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.opt, max_lr=0.001, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)

        self.scaler = GradScaler()

    def flatten(self):
        self.E.flatten_parameters()
        self.G.flatten_parameters()

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.args.dim_h * 2, self.args.nlayers)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data 

    def encode(self, input):
        #input = self.drop(self.embed(input)) #not use as there are denoising already
        if self.args.use_transformer:
            h=self.transformer(input)[0]

        else:
            input = self.embed(input)                         
            #_, (h, _) = self.E(input)
            #h = torch.cat([h[-2], h[-1]], 1)
            output, (final_hidden_state, final_cell_state) = self.E(input)
            if self.args.no_Attention:
                h = torch.cat([final_hidden_state[-2], final_hidden_state[-1]], 1)
            else:
                output = output.permute(1, 0, 2)
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
        #s = reparameterize(mu, logvar) #very bad
        if is_test == False:
            z = straightThrough(s, is_logit=True, stochastic=True)
        else:
            z = straightThrough(mu, is_logit=True, stochastic=False)
        return mu, logvar, z
    
    def encode2z(self, input, is_test=False):
        mu, logvar, u_pert = self.encode(input)
        #s = reparameterize_corr(mu, logvar, u_pert, self.args.device) ##worse
        s = reparameterize(mu, logvar)
        if is_test == False:
            z = s
        else: 
            z = mu
        
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

    def forward(self, inputs, is_test=False):
        if self.args.is_triplet:
            # if self.args.distance_type == 'hamming':
            #     mu, logvar, anchor = self.encode2binary(inputs, is_test)
            # else:
            #     mu, logvar, anchor = self.encode2z(inputs, is_test)
            losses, anchor = self.autoenc(inputs, is_test)
            return  losses, anchor.clone()
        else:
            _inputs = noisy(self.vocab, input, *self.args.noise) if is_test == False else inputs
            mu, logvar = self.encode(_inputs)
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, input)
            return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def loss(self, losses):
        if self.args.is_triplet:
            return losses['triplet']
        else:
            return losses['rec']

    def generate_triplet(self, inputs, is_test=False):
        mutation_rate = np.random.uniform(self.args.similar_noise, self.args.divergent_noise)
        used_margin = mutation_rate - self.args.similar_noise
        mutation_type = np.random.randint(4)
        similar_noises = [0,0,0,0]
        divergent_noises = [0,0,0,0]
        similar_noises[mutation_type] = self.args.similar_noise
        divergent_noises[mutation_type] = mutation_rate
        similar_inputs = noisy_dna(inputs, *similar_noises)
        divergent_inputs = noisy_dna(inputs, *divergent_noises)

        inputs = convert_sequences_to_kmers(inputs, self.args.k)
        similar_inputs = convert_sequences_to_kmers(similar_inputs, self.args.k)
        divergent_inputs = convert_sequences_to_kmers(divergent_inputs, self.args.k)

        inputs, _ = tokenization(inputs, self.vocab, self.args.device, self.args.seqlen - self.args.k + 1)
        similar_inputs, _ = tokenization(similar_inputs, self.vocab, self.args.device, self.args.seqlen - self.args.k + 1)
        divergent_inputs, _ = tokenization(divergent_inputs, self.vocab, self.args.device, self.args.seqlen - self.args.k + 1)
        
        # mu, logvar, anchor = self(inputs, is_test)
        if self.args.distance_type == 'hamming':
            mu, logvar, anchor = self.encode2binary(inputs, is_test)
            _, _, positive = self.encode2binary(similar_inputs, is_test)
            _, _, negative = self.encode2binary(divergent_inputs, is_test)
        else:
            mu, logvar, anchor = self.encode2z(inputs, is_test)
            _, _, positive = self.encode2z(similar_inputs, is_test)
            _, _, negative = self.encode2z(divergent_inputs, is_test)
        return mu, logvar, anchor, positive, negative, used_margin

    def autoenc(self, inputs, mu, logvar, anchor, is_test=False, is_evaluate=False):
        if self.args.is_triplet:
            mu, logvar, anchor, positive, negative, used_margin = self.generate_triplet(inputs, is_test)
            return {'triplet': self.triplet_loss(anchor, positive, negative, used_margin) if is_evaluate == False else self.triplet_loss_detail(anchor, positive, negative, used_margin)}, anchor.clone()
        else:
            inputs = convert_sequences_to_kmers(inputs, self.args.k)
            inputs, targets = tokenization(inputs, self.vocab, self.args.device, self.args.seqlen - self.args.k + 1)                                                                      
            _, _, _, logits = self(inputs, is_test)
            return {'rec': self.loss_rec(logits, targets).mean()}

    # def step(self, losses):
    #     self.opt.zero_grad()
    #     losses['loss'].backward()
    #     # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    #     #nn.utils.clip_grad_norm_(self.parameters(), clip)
    #     self.opt.step()
    #     #self.scheduler.step()
    
    def step(self, accelerator, losses, custom_lr=None):
        if custom_lr is not None:
            for param_group in self.opt.param_groups:
                param_group['lr'] = custom_lr

        self.opt.zero_grad()
        if self.args.use_amp:
            # self.scaler.scale(losses['loss']).backward()
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

    def triplet_loss(self, anchor, positive, negative, margin):           
        ###################### the triplet loss ###################################
        if self.args.distance_type == 'hamming':
            pos_dist = self.hamming_distance(anchor, positive)
            neg_dist = self.hamming_distance(anchor, negative)
            tripletLoss = torch.clamp(margin * self.args.lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12).mean() 
        elif self.args.distance_type == "euclidean":
            pos_dist = self.euclidean_distance(anchor, positive)
            neg_dist = self.euclidean_distance(anchor, negative)
            tripletLoss = torch.clamp(margin * self.args.lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12).mean() 
        else:
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
            tripletLoss = torch.clamp(margin * self.args.lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12).mean()
            #tripletLoss = self.cosine_margin_triplet_loss(anchor, positive, negative, margin)
        return tripletLoss
    
    def triplet_loss_detail(self, anchor, positive, negative, margin):           
        ###################### the triplet loss ###################################
        if self.args.distance_type == 'hamming':
            pos_dist = self.hamming_distance(anchor, positive)
            neg_dist = self.hamming_distance(anchor, negative)
            tripletLoss = torch.clamp(margin * self.args.lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12)
        elif self.args.distance_type == "euclidean":
            pos_dist = self.euclidean_distance(anchor, positive)
            neg_dist = self.euclidean_distance(anchor, negative)
            tripletLoss = torch.clamp(margin * self.args.lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12)
        else:
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
            tripletLoss = torch.clamp(margin * self.args.lambda_margin + pos_dist - neg_dist + self.args.lambda_sim * pos_dist, min=1e-12)
            #tripletLoss = self.cosine_margin_triplet_loss(anchor, positive, negative, margin)
        return tripletLoss

    def cosine_margin_triplet_loss(self, anchor, positive, negative, margin):           
        ###################### the triplet loss ###################################
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative) 
        pos_entropy = torch.exp(pos_sim / self.args.lambda_margin - margin)
        neg_entropy = torch.exp(neg_sim / self.args.lambda_margin)
        #return torch.clamp(-torch.log(pos_entropy / (pos_entropy + neg_entropy)), min=1e-12).mean()
        return torch.clamp(-torch.log(pos_entropy / neg_entropy), min=1e-12).mean() 

class VAE(DAE):
    """Variational Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)

    def loss(self, losses):
        if self.args.is_triplet:
            return losses['triplet'] + self.args.lambda_kl * losses['kl']
        else:
            return losses['rec'] + self.args.lambda_kl * losses['kl']

    def autoenc(self, inputs, targets, is_test=False, is_evaluate=False):
        if self.args.is_triplet:
            mu, logvar, anchor, positive, negative, used_margin = self.generate_triplet(inputs, is_test)
            return {'triplet': self.triplet_loss(anchor, positive, negative, used_margin) if is_evaluate == False else self.triplet_loss_detail(anchor, positive, negative, used_margin),
                    'kl': loss_kl(mu, logvar)}, anchor.clone()
        else:
            inputs = convert_sequences_to_kmers(inputs, self.args.k)
            inputs, targets = tokenization(inputs, self.vocab, self.args.device, self.args.seqlen - self.args.k + 1) 
            mu, logvar, _, logits = self(inputs, is_test)
            return {'rec': self.loss_rec(logits, targets).mean(),
                    'kl': loss_kl(mu, logvar)}
               
class AAE(DAE):
    """Adversarial Auto-Encoder"""

    def __init__(self, vocab, args):
        super().__init__(vocab, args)
        self.D = nn.Sequential(nn.Linear(args.dim_z, args.dim_d), nn.ReLU(),
            nn.Linear(args.dim_d, 1), 
            # nn.Sigmoid()
            )
        self.optD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0.5, 0.999))

        self.scalerD = GradScaler() # scaler for discriminator

    def loss_adv(self, z):
        zn = torch.randn_like(z)
        zeros = torch.zeros(len(z), 1, device=z.device)
        ones = torch.ones(len(z), 1, device=z.device)
        loss_d = F.binary_cross_entropy_with_logits(self.D(z.detach()), zeros) + \
            F.binary_cross_entropy_with_logits(self.D(zn), ones)
        loss_g = F.binary_cross_entropy_with_logits(self.D(z), ones)
        return loss_d, loss_g

    def loss(self, losses):
        if self.args.is_triplet:
            return losses['triplet'] + self.args.lambda_adv * losses['adv'] + \
                self.args.lambda_p * losses['|lvar|']
        else:
            return losses['rec'] + self.args.lambda_adv * losses['adv'] + \
                self.args.lambda_p * losses['|lvar|']
        
    def forward(self, inputs, is_test=False, is_D=False):
        if self.args.is_triplet:
            if is_D==False:
                # if self.args.distance_type == 'hamming':
                #     mu, logvar, anchor = self.encode2binary(inputs, is_test)
                # else:
                #     mu, logvar, anchor = self.encode2z(inputs, is_test)
                
                losses, anchor = self.autoenc(inputs, is_test)
                return  losses, anchor.clone()
            else:
                loss_d, adv = self.loss_adv(inputs)
                return loss_d
        else:
            _inputs = noisy(self.vocab, input, *self.args.noise) if is_test == False else inputs
            mu, logvar = self.encode(_inputs)
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, input)
            return mu, logvar, z, logits
        
    def autoenc(self, inputs, is_test=False, is_evaluate=False):
        if self.args.is_triplet:
            mu, logvar, anchor, positive, negative, used_margin = self.generate_triplet(inputs, is_test)
            adv_z = anchor if is_test == False else mu
            loss_d, adv = self.loss_adv(adv_z)
            #loss_d_p, adv_p = self.loss_adv(positive) 
            #loss_d_n, adv_n = self.loss_adv(negative)
            return {'triplet': self.triplet_loss(anchor, positive, negative, used_margin) if is_evaluate == False else self.triplet_loss_detail(anchor, positive, negative, used_margin),
                'adv': adv,
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d}, anchor.clone()
        else:
            inputs = convert_sequences_to_kmers(inputs, self.args.k)
            inputs, targets = tokenization(inputs, self.vocab, self.args.device, self.args.seqlen - self.args.k + 1)        
            _, logvar, z, logits = self(inputs, is_test)
            loss_d, adv = self.loss_adv(z)
            return {'rec': self.loss_rec(logits, targets).mean(),
                'adv': adv,
                '|lvar|': logvar.abs().sum(dim=1).mean(),
                'loss_d': loss_d}

    def step(self, accelerator, losses, custom_lr=None, is_D=False):
        if is_D == False:
            super().step(accelerator, losses, custom_lr)
        else:
            if custom_lr is not None:
                for param_group in self.optD.param_groups:
                    param_group['lr'] = custom_lr
            self.optD.zero_grad()
            if self.args.use_amp:
                accelerator.backward(self.scalerD.scale(losses['loss_d']))
                self.scalerD.step(self.optD)
                self.scalerD.update()
            else:
                accelerator.backward(losses['loss_d'])
                self.optD.step()
class TransformerEncoder(nn.Module):
    """
    Encoder-only Transformer model.
    """

    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        dropout_p,
        abs_position_encoding=False,
    ):
        super().__init__()

        # INFO
        self.dim_model = dim_model

        # LAYERS
        # self.positional_encoder = PositionalEncoding(
        #     dim_model=dim_model, dropout_p=dropout_p, max_len=1000
        # )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=0,  # No decoder layers
            dropout=dropout_p,
        )
        self.pos_embedding = nn.Embedding(65, dim_model)
        self.dropout = nn.Dropout(dropout_p)
        self.LayerNorm = nn.LayerNorm(dim_model, eps=1e-8)
        self.position_encoder = PositionalEncoding(dim_model, dropout_p, max_len=1000)
        self.abs_position_encoding = abs_position_encoding
    def forward(self, src):
        # Src size must be (batch_size, src sequence length)
        src = src.permute(1, 0)
        if self.abs_position_encoding:
            src = self.embedding(src) * math.sqrt(self.dim_model)
            src = self.position_encoder(src)
        else:
            seq_length = src.size(1)
            # print(seq_length)
            position_ids = torch.arange(0,seq_length, dtype=torch.long,device = src.device)
            position_ids = position_ids.unsqueeze(0).expand_as(src)
            # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
            embed = self.embedding(src) * math.sqrt(self.dim_model)
            pos = self.pos_embedding(position_ids)
            src = embed + pos
        src = self.dropout(src)
        # Permute to shape (sequence length, batch_size, dim_model)
        src = src.permute(1, 0, 2)

        # Transformer encoder blocks - Out size = (sequence length, batch_size, dim_model)
        encoder_output = self.transformer.encoder(src)
        # Permute back to shape (batch_size, sequence length, dim_model) for output
        encoder_output = encoder_output.permute(1, 0, 2)

        return encoder_output

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.dim_model = dim_model
        self.max_len = max_len
        
    def forward(self, token_embedding):
        # Calculate positional encodings on the fly
        position = torch.arange(0, self.max_len).unsqueeze(1).to(token_embedding.device).float()
        div_term = torch.exp(torch.arange(0, self.dim_model, 2).float() * (-math.log(10000.0) / self.dim_model)).to(token_embedding.device)
        
        pe = torch.zeros(self.max_len, self.dim_model).to(token_embedding.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Only use the required portion of the positional encoding matrix
        pos_encoding_cloned = pe[:token_embedding.size(0), :].clone()
        token_embedding = token_embedding + pos_encoding_cloned
        return token_embedding