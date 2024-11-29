import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from image_synthesis.modeling.utils.misc import distributed_sinkhorn
from image_synthesis.distributed.distributed import all_reduce

def value_scheduler(init_value, dest_value, step, step_range, total_steps, scheduler_type='cosine'):
    assert scheduler_type in ['cosine', 'step'], 'scheduler {} not implemented!'.format(scheduler_type)

    step_start, step_end = tuple(step_range)
    if step_end <= 0:
        step_end = total_steps

    if step < step_start:
        return init_value
    if step > step_end:
        return dest_value

    factor = float(step - step_start) / float(max(1, step_end - step_start))
    if scheduler_type == 'cosine':
        factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * factor)))
    elif scheduler_type == 'step':
        factor = 1 - factor
    else:
        raise NotImplementedError('scheduler type {} not implemented!'.format(scheduler_type))
    if init_value >= dest_value: # decrease
        value = dest_value + (init_value - dest_value) * factor
    else: # increase
        factor = 1 - factor
        value = init_value + (dest_value - init_value) * factor
    return value 

def gumbel_softmax(logits, temperature=1.0, gumbel_scale=1.0, dim=-1, hard=True):
    # gumbels = torch.distributions.gumbel.Gumbel(0,1).sample(logits.shape).to(logits)
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    # adjust the scale of gumbel noise
    gumbels = gumbels * gumbel_scale

    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

# class for quantization
class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, 
                n_e, 
                e_dim, 
                beta=0.25,
                masked_embed_start=None,
                embed_init_scale=1.0,
                embed_ema=False,
                get_embed_type='matmul',
                distance_type='euclidean',

                gumbel_sample=False,
                adjust_logits_for_gumbel='sqrt',
                gumbel_sample_stop_step=None,
                temperature_step_range=(0,15000),
                temperature_scheduler_type='cosine',
                temperature_init=1.0,
                temperature_dest=1/16.0,
                gumbel_scale_init=1.0,
                gumbel_scale_dest=1.0,
                gumbel_scale_step_range=(0,1),
                gumbel_scale_scheduler_type='cosine',
        ):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embed_ema = embed_ema
        self.gumbel_sample = gumbel_sample
        self.adjust_logits_for_gumbel = adjust_logits_for_gumbel
        self.temperature_step_range = temperature_step_range
        self.temperature_init = temperature_init
        self.temperature_dest = temperature_dest
        self.temperature_scheduler_type = temperature_scheduler_type
        self.gumbel_scale_init = gumbel_scale_init
        self.gumbel_scale_dest = gumbel_scale_dest 
        self.gumbel_scale_step_range = gumbel_scale_step_range
        self.gumbel_sample_stop_step = gumbel_sample_stop_step
        self.gumbel_scale_scheduler_type = gumbel_scale_scheduler_type
        if self.gumbel_sample_stop_step is None:
            self.gumbel_sample_stop_step = max(self.temperature_step_range[-1], self.temperature_step_range[-1])

        self.get_embed_type = get_embed_type
        self.distance_type = distance_type

        if self.embed_ema:
            self.decay = 0.99
            self.eps = 1.0e-5
            embed = torch.randn(n_e, e_dim)
            # embed = torch.zeros(n_e, e_dim)
            # embed.data.uniform_(-embed_init_scale / self.n_e, embed_init_scale / self.n_e)
            self.register_buffer("embedding", embed)
            self.register_buffer("cluster_size", torch.zeros(n_e))
            self.register_buffer("embedding_avg", embed.clone())
        else:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-embed_init_scale / self.n_e, embed_init_scale / self.n_e)

        self.masked_embed_start = masked_embed_start
        if self.masked_embed_start is None:
            self.masked_embed_start = self.n_e

        if self.distance_type == 'learned':
            self.distance_fc = nn.Linear(self.e_dim, self.n_e)
    @property
    def device(self):
        if isinstance(self.embedding, nn.Embedding):
            return self.embedding.weight.device
        return self.embedding.device

    @property
    def norm_feat(self):
        return self.distance_type in ['cosine', 'sinkhorn']
    
    @property
    def embed_weight(self):
        if isinstance(self.embedding, nn.Embedding):
            return self.embedding.weight
        else:
            return self.embedding
    
    def get_codebook(self):
        codes = {
            'default': {
                'code': self.embedding
            }
        }

        if self.masked_embed_start < self.n_e:
            codes['unmasked'] = {'code': self.embedding[:self.masked_embed_start]}
            codes['masked'] = {'code': self.embedding[self.masked_embed_start:]}

            default_label = torch.ones((self.n_e)).to(self.device)
            default_label[self.masked_embed_start:] = 0
            codes['default']['label'] = default_label
        return codes

    def norm_embedding(self):
        if self.training:
            with torch.no_grad():
                w = self.embed_weight.data.clone()
                w = F.normalize(w, dim=1, p=2)
                if isinstance(self.embedding, nn.Embedding):
                    self.embedding.weight.copy_(w)
                else:
                    self.embedding.copy_(w)


    def get_index(self, logits, topk=1, step=None, total_steps=None):
        """
        logits: BHW x N
        topk: the topk similar codes to be sampled from

        return:
            indices: BHW
        """
        
        if self.gumbel_sample:
            gumbel = True
            if self.training:
                if step > self.gumbel_sample_stop_step and self.gumbel_sample_stop_step > 0:
                    gumbel = False
            else:
                gumbel = False
        else:
            gumbel = False

        if gumbel:
            temp = value_scheduler(init_value=self.temperature_init,
                                    dest_value=self.temperature_dest,
                                    step=step,
                                    step_range=self.temperature_step_range,
                                    total_steps=total_steps,
                                    scheduler_type=self.temperature_scheduler_type
                                    )
            scale = value_scheduler(init_value=self.gumbel_scale_init,
                                    dest_value=self.gumbel_scale_dest,
                                    step=step,
                                    step_range=self.gumbel_scale_step_range,
                                    total_steps=total_steps,
                                    scheduler_type=self.gumbel_scale_scheduler_type
                                    )
            if self.adjust_logits_for_gumbel == 'none':
                pass
            elif self.adjust_logits_for_gumbel == 'sqrt':
                logits = torch.sqrt(logits)
            elif self.adjust_logits_for_gumbel == 'log':
                logits = torch.log(logits)
            else:
                raise NotImplementedError
            
            # for logits, the larger the value is, the corresponding code shoule not be sampled, so we need to negative it
            logits = -logits
            # one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=True) # BHW x N
            logits = gumbel_softmax(logits, temperature=temp, gumbel_scale=scale, dim=1, hard=True)
        else:
            logits = -logits
        
        # now, the larger value should be sampled
        if topk == 1:
            indices = torch.argmax(logits, dim=1)
        else:
            assert not gumbel, 'For gumbel sample, topk may introduce some random choices of codes!'
            topk = min(logits.shape[1], topk)

            _, indices = torch.topk(logits, dim=1, k=topk) # N x K
            chose = torch.randint(0, topk, (indices.shape[0],)).to(indices.device) # N
            chose = torch.zeros_like(indices).scatter_(1, chose.unsqueeze(dim=1), 1.0) # N x K
            indices = (indices * chose).sum(dim=1, keepdim=False)
            
            # filtered_logits = logits_top_k(logits, filter_ratio=topk, minimum=1, filter_type='count')
            # probs = F.softmax(filtered_logits * 1, dim=1)
            # indices = torch.multinomial(probs, 1).squeeze(dim=1) # BHW
            
        return indices

    def get_distance(self, z, code_type='all'):
        """
        z: L x D, the provided features

        return:
            d: L x N, where N is the number of tokens, the smaller distance is, the more similar it is
        """
        if self.distance_type == 'euclidean':
            d = torch.sum(z ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embed_weight**2, dim=1) - 2 * \
                torch.matmul(z, self.embed_weight.t())
        elif self.distance_type == 'learned':
            d = 0 - self.distance_fc(z) # BHW x N
        elif self.distance_type == 'sinkhorn':
            s = torch.einsum('ld,nd->ln', z, self.embed_weight) # BHW x N
            d = 0 - distributed_sinkhorn(s.detach())
            # import pdb; pdb.set_trace()
        elif self.distance_type == 'cosine':
            d = 0 - torch.einsum('ld,nd->ln', z, self.embed_weight) # BHW x N
        else:
            raise NotImplementedError('distance not implemented for {}'.format(self.distance_type))
        
        if code_type == 'masked':
            d = d[:, self.masked_embed_start:]
        elif code_type == 'unmasked':
            d = d[:, :self.masked_embed_start]

        return d

    def _quantize(self, z, token_type=None, topk=1, step=None, total_steps=None):
        """
            z: L x D
            token_type: L, 1 denote unmasked token, other masked token
        """
        d = self.get_distance(z)

        # find closest encodings 
        # import pdb; pdb.set_trace()
        if token_type is None or self.masked_embed_start == self.n_e:
            # min_encoding_indices = torch.argmin(d, dim=1) # L
            min_encoding_indices = self.get_index(d, topk=topk, step=step, total_steps=total_steps)
        else:
            min_encoding_indices = torch.zeros(z.shape[0]).long().to(z.device)
            idx = token_type == 1
            if idx.sum() > 0:
                d_ = d[idx][:, :self.masked_embed_start] # l x n
                # indices_ = torch.argmin(d_, dim=1)
                indices_ = self.get_index(d_, topk=topk, step=step, total_steps=total_steps)
                min_encoding_indices[idx] = indices_
            idx = token_type != 1
            if idx.sum() > 0:
                d_ = d[idx][:, self.masked_embed_start:] # l x n
                # indices_ = torch.argmin(d_, dim=1)
                indices_ = self.get_index(d_, topk=topk, step=step, total_steps=total_steps)
                indices_ += self.masked_embed_start
                min_encoding_indices[idx] = indices_

        if self.get_embed_type == 'matmul':
            min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
            min_encodings.scatter_(1, min_encoding_indices.unsqueeze(1), 1)
            # import pdb; pdb.set_trace()
            z_q = torch.matmul(min_encodings, self.embed_weight)#.view(z.shape)
        elif self.get_embed_type == 'retrive':
            z_q = F.embedding(min_encoding_indices, self.embed_weight)#.view(z.shape)
        else:
            raise NotImplementedError

        return z_q, min_encoding_indices

    def forward(self, z, token_type=None, topk=1, step=None, total_steps=None):
        """
            z: B x C x H x W
            token_type: B x 1 x H x W
        """
        if self.distance_type in ['sinkhorn', 'cosine']:
            # need to norm feat and weight embedding    
            self.norm_embedding()            
            z = F.normalize(z, dim=1, p=2)

        # reshape z -> (batch, height, width, channel) and flatten
        batch_size, _, height, width = z.shape
        # import pdb; pdb.set_trace()
        z = z.permute(0, 2, 3, 1).contiguous() # B x H x W x C
        z_flattened = z.view(-1, self.e_dim) # BHW x C
        if token_type is not None:
            token_type_flattened = token_type.view(-1)
        else:
            token_type_flattened = None

        z_q, min_encoding_indices = self._quantize(z_flattened, token_type=token_type_flattened, topk=topk, step=step, total_steps=total_steps)
        z_q = z_q.view(batch_size, height, width, -1) #.permute(0, 2, 3, 1).contiguous()

        if self.training and self.embed_ema:
            # import pdb; pdb.set_trace()
            assert self.distance_type in ['euclidean', 'cosine']
            indices_onehot = F.one_hot(min_encoding_indices, self.n_e).to(z_flattened.dtype) # L x n_e
            indices_onehot_sum = indices_onehot.sum(0) # n_e
            z_sum = (z_flattened.transpose(0, 1) @ indices_onehot).transpose(0, 1) # n_e x D

            all_reduce(indices_onehot_sum)
            all_reduce(z_sum)

            self.cluster_size.data.mul_(self.decay).add_(indices_onehot_sum, alpha=1 - self.decay)
            self.embedding_avg.data.mul_(self.decay).add_(z_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n
            embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)

        if self.embed_ema:
            loss = (z_q.detach() - z).pow(2).mean()
        else:
            # compute loss for embedding
            loss = torch.mean((z_q.detach()-z).pow(2)) + self.beta * torch.mean((z_q - z.detach()).pow(2))

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        unique_idx = min_encoding_indices.unique()
        output = {
            'quantize': z_q,
            'used_unmasked_quantize_embed': torch.zeros_like(loss) + (unique_idx<self.masked_embed_start).sum(),
            'used_masked_quantize_embed': torch.zeros_like(loss) + (unique_idx>=self.masked_embed_start).sum(),
            'quantize_loss': loss,
            'index': min_encoding_indices.view(batch_size, height, width)
        }
        if token_type_flattened is not None:
            unmasked_num_token = all_reduce((token_type_flattened == 1).sum())
            masked_num_token = all_reduce((token_type_flattened != 1).sum())
            output['unmasked_num_token'] = unmasked_num_token
            output['masked_num_token'] = masked_num_token

        return output

    def get_codebook_entry(self, indices, shape):
        # import pdb; pdb.set_trace()

        # shape specifying (batch, height, width)
        if self.get_embed_type == 'matmul':
            min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
            min_encodings.scatter_(1, indices[:,None], 1)
            # get quantized latent vectors
            z_q = torch.matmul(min_encodings.float(), self.embed_weight)
        elif self.get_embed_type == 'retrive':
            z_q = F.embedding(indices, self.embed_weight)
        else:
            raise NotImplementedError

        # import pdb; pdb.set_trace()
        if shape is not None:
            z_q = z_q.view(*shape, -1) # B x H x W x C

            if len(z_q.shape) == 4:
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

