"""
This transformer model is for PUT in CVPR 2022
"""
import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F

from image_synthesis.modeling.utils.misc import get_token_type
from image_synthesis.distributed.distributed import get_local_rank
from image_synthesis.modeling.utils.misc import logits_top_k

from image_synthesis.modeling.vector_predictor.transformer_block import Block

class MaskedImageInpaintingTransformer(nn.Module):
    def __init__(
        self,
        *,
        n_layer, # number of layers in transformer
        content_seq_len, # length of content sequences
        embd_pdrop=0., # embedding dropout prob

        n_embd, # the embed dim
        n_head, # the number of heads
        attn_pdrop=0.1, # attention dropout prob
        resid_pdrop=0.1, # residual attention dropout prob
        block_activate='GELU',
        mlp_type='linear', # linear mlp or conv mlp
        mlp_hidden_times=4, # the times of hidden dimension in the MLP of attetntion block

        attn_content_with_mask=False,

        content_ignore_token=-100,

        ckpt_path=None,
        ignore_keys=[],

        input_feature_type='origin',
        # args for training
        weight_decay=0.01,
        random_quantize=0.2, # random quantize the feature, only when the input feature is not quantized
        num_token=None,
        
        embed_dim=256,

        trainable=True
    ):
        super().__init__()

        self.trainable = trainable
        
        self.attn_content_with_mask = attn_content_with_mask

        self.emb_proj = nn.Linear(embed_dim, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, content_seq_len, n_embd))
        
        # drop for embedding
        if embd_pdrop > 0:
            self.drop = nn.Dropout(embd_pdrop)
        else:
            self.drop = None
        
        # transformer
        self.blocks = nn.Sequential(*[Block(
                n_embd=n_embd,
                n_head=n_head,
                seq_len=content_seq_len,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                causal=False,
                mlp_type=mlp_type,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for n in range(n_layer)])

        # final prediction head
        # out_cls = self.content_codec.get_number_of_tokens() if num_token is None else 
        out_cls = num_token
        self.layer_norm = nn.LayerNorm(n_embd)
        self.to_logits = nn.Linear(n_embd, out_cls)
        
        self.content_seq_len = content_seq_len
        self.content_ignore_token = content_ignore_token
        self.input_feature_type = input_feature_type
        self.weight_decay = weight_decay
        self.random_quantize = random_quantize

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            self.apply(self._init_weights)
            
        self._set_trainable()

    def _set_trainable(self):
        if not self.trainable:
            for pn, p in self.named_parameters():
                p.requires_grad = False
            self.eval()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if 'model' in sd:
            sd = sd['model']
        else:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Transformer: Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print("Transformer: Load pretrained model from {}".format(path))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)         

    @property
    def device(self):
        return self.to_logits.weight.device

    def get_sample_token(self, emb, mask_=None, filter_ratio=50, num_token_per_iter=-1, pos_embedding=True, return_logits=False):
        """
        emb : B x HW x D

        return:
            sample: sampled token, B x HW
        """
        
        if pos_embedding:
            emb = self.emb_proj(emb) # B x HW x D
            if self.pos_emb is not None:
                emb = emb + self.pos_emb

        for block_idx in range(len(self.blocks)):   
            emb, _ = self.blocks[block_idx](emb, mask=None) # B x HW x D, B x HW x HW
        
        emb = self.layer_norm(emb)
        logits = self.to_logits(emb) # B x HW x n

        if return_logits:
            return logits

        other_out = {}

        if num_token_per_iter == -1 or num_token_per_iter >= self.content_seq_len:
            filtered_logits = logits_top_k(logits, filter_ratio=50, minimum=1, filter_type='count') # b x HW x C
            probs = F.softmax(filtered_logits * 1, dim = -1) # b x num x C
            sample = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(*probs.shape[:2]) # b x HW

            return sample, True, other_out

        # for each sample, get the max
        logits_, _ = logits.max(dim=-1) # b x HW
        logits_.masked_fill_(mask_, float('-inf'))

        _, index = torch.topk(logits_, dim=1, k=num_token_per_iter) # b x num, in range [0, HW)
        index_one_hot = F.one_hot(index, logits.shape[1]) # b x num x HW
        logits = torch.einsum('blc,bnl->bnc', logits, index_one_hot.float()) # b x n x C

        # sample
        filtered_logits = logits_top_k(logits, filter_ratio=filter_ratio, minimum=1, filter_type='count') # b x num x C
        probs = F.softmax(filtered_logits * 1, dim = -1) # b x num x C
        sample = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(*probs.shape[:2]) # b x num

        other_out['index'] = index
        other_out['index_one_hot'] = index_one_hot

        return sample, False, other_out

    @torch.no_grad()
    def pos_embedding(self, feature, index_one_hot=None):
        feature = self.emb_proj(feature) # b x num x D
        pos_emb = self.pos_emb
        if index_one_hot != None:
            if self.pos_emb.shape[0] == index_one_hot.shape[0]:
                pos_emb = torch.einsum('bld,bnl->bnd', self.pos_emb, index_one_hot.float()) # b x num x d
            else:
                pos_emb = self.pos_emb.repeat(index_one_hot.shape[0], 1, 1)
                pos_emb = torch.einsum('bld,bnl->bnd', pos_emb, index_one_hot.float()) # b x num x d
        feature += pos_emb # b x num x D

        return feature

    @torch.no_grad()
    def sample( # sample_raster
        self,
        *,
        batch,
        filter_ratio = 0.8,
        filter_type = 'count',
        temperature = 1.0,
        only_masked=True,
        with_process_bar=False,
        return_gt=True,
        return_mask_gt=True,
        return_reconstruction=True,
        mask_low_to_high=False,
        num_token_per_iter=1,
        **kwargs,
    ): 

        if num_token_per_iter != 1:
            raise NotImplementedError

        self.eval()

        for k in batch.keys():
            if torch.is_tensor(batch[k]):# isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)

        if mask_low_to_high:
            low_res = self.content_codec.token_shape
            ori_res = batch['mask'].shape[-2:]
            assert low_res is not None 
            # import pdb; pdb.set_trace()
            mask_ = F.interpolate(batch['mask'].float(), size=low_res, mode='nearest')
            mask_ = F.interpolate(mask_, size=ori_res, mode='nearest').bool()            
        else:
            mask_ = batch['mask']

        data_mask = self.content_codec.get_features(batch['image'], 
                                                    mask=mask_, 
                                                    return_quantize_feature=self.input_feature_type == 'quantized',
                                                    return_token=True) # B x C x H x W
        if self.input_feature_type == 'origin':
            feat_mask = data_mask['feature'] # B x C x H x W
        elif self.input_feature_type == 'quantized':
            feat_mask = data_mask['feature_quantize'] # B x C x H x W
        else:
            raise NotImplementedError('inpute feature type {} not implemented!'.format(self.input_feature_type))
            
        b, _, h, w = feat_mask.shape
        feat_mask = self.emb_proj(feat_mask.permute(0, 2, 3, 1).view(b, h*w, -1).contiguous()) # B x HW x D
        # NOTE: the feature for masked tokens are the same?
        if self.pos_emb is not None:
            feat_mask = feat_mask + self.pos_emb

        token_type = get_token_type(mask_, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H x W
    
        content_feat = feat_mask
        content_token = data_mask['token'].view(b, -1)
        content_mask = (token_type == 1).view(b, -1)

        # import pdb; pdb.set_trace()
        bar = range(0, self.content_seq_len)
        if with_process_bar:
            bar = tqdm(bar, position=get_local_rank(), desc='Local rank: {}'.format(get_local_rank()))
        for i in bar:
            start_ = i 
            end_ = i+1
            itr_cont_mask = content_mask[:, start_:end_] # B x 1

            if only_masked:
                pred = itr_cont_mask.sum() < b # if there are some masked 
            else:
                pred = True
            
            if pred: 
                emb = content_feat
                if self.drop is not None:
                    emb = self.drop(emb)
                if self.attn_content_with_mask:
                    attn_mask = content_mask
                else:
                    attn_mask = None
                for block_idx in range(len(self.blocks)):   
                    emb, att_weight = self.blocks[block_idx](emb, mask=attn_mask) # B x HW x D, B x HW x HW
                
                # 3) get logits
                emb = emb[:, start_:end_]
                emb = self.layer_norm(emb)
                logits = self.to_logits(emb) # B x 1 x C

                # sample
                filtered_logits = logits_top_k(logits, filter_ratio=filter_ratio, minimum=1, filter_type=filter_type) # B x 1 x C
                probs = F.softmax(filtered_logits * temperature, dim = -1) # B x 1 x C
                sample = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(b, 1) # B x 1

                # change contents
                sample_feat = self.content_codec.get_codebook_entry_with_token(sample)['feature'] # B x 1 x C
                sample_feat = self.emb_proj(sample_feat) # B x 1 x D
                if self.pos_emb is not None:
                    sample_feat += self.pos_emb[:, start_:end_,:]
                # import pdb; pdb.set_trace()
                content_feat[:, start_:end_] = sample_feat * (1 - itr_cont_mask.to(sample_feat.dtype).unsqueeze(dim=-1)) + content_feat[:, start_:end_] * itr_cont_mask.to(sample_feat.dtype).unsqueeze(dim=-1)
                content_token[:, start_:end_][~itr_cont_mask] = sample[~itr_cont_mask]
                content_mask[:, start_:end_] = True
                assert content_mask[:, :end_].sum() == b * end_ # make sure that previous content tokens are all unmasked
        
        # decode
        masked_im = batch['image'] * batch['mask']
        sampled_im = self.content_codec.decode(content_token, mask_im=masked_im, mask=batch['mask'], token_shape=[h,w])

        output = {
            'completed': sampled_im
        }
        if return_gt:
            output['input'] = batch['image']
        if return_mask_gt:
            output['mask_input'] = masked_im
        if return_reconstruction:
            token = self.content_codec.get_tokens(batch['image'], mask=batch['mask'])
            output['reconstruction'] = self.content_codec.decode(token['token'], token_shape=[h,w])
        self.train()
        return output

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d) #TODO(torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    if not p.requires_grad:
                        continue 

                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

            no_decay.add('pos_emb')

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def prepare_data(self, image, mask, data_mask):
        """
        Get the feature from image

        Args:
            image: B x 3 x H x W
            mask: B x 1 x H x W
        """

        if self.input_feature_type == 'origin':
            feat_mask = data_mask['feature'] # B x C x H x W
            b, _, h, w = feat_mask.shape
            # random change origin feature with quantized feature
            token_type = get_token_type(mask, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H x W
            valid_token_mask = token_type == 1
            if self.random_quantize > 0:
                quantize_mask = torch.rand(*token_type.shape).to(token_type.device) < self.random_quantize # B x 1 x H x W, in range [0, 1)
                # only quantize those unmasked tokens
                quantize_mask = (valid_token_mask * quantize_mask).to(feat_mask.dtype) # 1 denotes to be quantized
                feat_mask = feat_mask * (1-quantize_mask) +  (data_mask['feature_quantize'] * quantize_mask) # B x C x H x W
        elif self.input_feature_type == 'quantized':
            feat_mask = data_mask['feature_quantize'] # B x C x H x W
            b, _, h, w = feat_mask.shape
            token_type = get_token_type(mask, type='pixel_shuffle', token_shape=[h,w]) # B x 1 x H x W
            valid_token_mask = token_type == 1
        else:
            raise NotImplementedError('input feature type {} is not impleted!'.format(self.input_feature_type))

        gt_features = data_mask['gt_features']
        token_target = gt_features['token'].view(b, h*w).contiguous() # B x HW
        feat_mask = self.emb_proj(feat_mask.permute(0, 2, 3, 1).view(b, h*w, -1).contiguous()) # B x HW x D
        if self.pos_emb is not None:
            feat_mask = feat_mask + self.pos_emb
        
        output = {
            'token_target': token_target, # B x HW
            'token_type': token_type.view(b, -1).contiguous(),
            'embedding': feat_mask,
            'token_shape': [b,h,w]
        }

        return output

    def forward(
            self, 
            batch,
            data_mask,
            return_loss=False, 
            return_logits=True, 
            return_att_weight=False,
            **kwargs):

        # 1) get data from input data
        data = self.prepare_data(batch['image'], mask=batch['mask'], data_mask=data_mask)
        emb = data['embedding']

        for block_idx in range(len(self.blocks)):   
            emb, att_weight = self.blocks[block_idx](emb, mask=None) # B x HW x D, B x HW x HW
        
        # 3) get logits
        emb = self.layer_norm(emb)
        logits = self.to_logits(emb) # B x HW x n

        # 4) get output, especially loss
        out = {}

        if return_logits:
            out['logits'] = logits
        if return_att_weight:
            out['attention_weight'] = att_weight

        if return_loss:
            token_target = data['token_target']
            token_type = data['token_type']
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), token_target.view(-1), ignore_index=self.content_ignore_token, reduction='none')
            loss_mask = (token_type!=1).to(loss).view(-1)
            loss = loss * loss_mask
            loss = torch.sum(loss) / (torch.sum(loss_mask) + 1e-18)
            out['loss'] = loss

            _, index = torch.topk(logits.view(-1, logits.shape[-1]), dim=1, k=1)
            index = index.view(-1) * loss_mask
            correct = (index == token_target.view(-1))
            correct = correct.to(loss).mean()
            out['accuracy'] = correct
        return out
