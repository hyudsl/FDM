import math

import torch
from torch import nn
import torch.nn.functional as F

class FullAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 seq_len=None, # the max length of sequence
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 causal=True,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.causal = causal

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len))
                                        .view(1, 1, seq_len, seq_len))


    def forward(self, x, mask=None):
        """
        x: B x T x C
        mask: None or tensor B x T, bool type. For values with False, no attention should be attened
        """
        B, T, C = x.size()
        # import pdb; pdb.set_trace()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        
        # print(q.shape, k.shape)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        if self.causal:
            # print(att.shape, self.mask.shape, T)
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        if mask is not None:
            mask = mask.view(B, 1, 1, T)
            att = att.masked_fill(~mask, float('-inf'))

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class ConvMLP(nn.Module):
    def __init__(
        self,
        n_embd,
        mlp_hidden_times,
        act, 
        resid_pdrop,
        spatial_size=None # (h, w) of input shape 
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_embd, out_channels=mlp_hidden_times*n_embd, kernel_size=3, stride=1,padding=1)
        self.act = act 
        self.conv2 = nn.Conv2d(in_channels=mlp_hidden_times*n_embd, out_channels=n_embd, kernel_size=3, stride=1,padding=1)
        self.dropout = nn.Dropout(resid_pdrop)
        self.spatial_size = spatial_size

    def forward(self, x):
        """
        x: B x T x C
        """
        # import pdb; pdb.set_trace()
        if self.spatial_size is None:
            length = x.shape[1]
            h = int(math.sqrt(length))
            w = h 
        else:
            h, w = self.spatial_size[0], self.spatial_size[1]
        x = x.view(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2) # B x C x H x W
        
        x = self.conv2(self.act(self.conv1(x)))
        x = x.permute(0, 2, 3, 1).view(x.shape[0], h*w, -1) # B x L x C
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self,
                 n_embd,
                 n_head,
                 seq_len,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 causal=True,
                 mlp_type='linear',
                 mlp_hidden_times=4,
                 activate='GELU',
                 ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            seq_len=seq_len,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            causal=causal
        )
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        if mlp_type == 'linear':
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )
        elif mlp_type == 'conv':
            self.mlp = ConvMLP(
                n_embd=n_embd,
                mlp_hidden_times=mlp_hidden_times,
                act=act, 
                resid_pdrop=resid_pdrop
            )

    def forward(self, x, mask=None):    
        a, att = self.attn(self.ln1(x), mask=mask)
        x = x + a 
        x = x + self.mlp(self.ln2(x))

        return x, att
