import torch
import torch.nn.functional as F

from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.modeling.codecs.base_codec import BaseCodec
from image_synthesis.modeling.utils.misc import get_token_type


class PatchVQGAN(BaseCodec):
    def __init__(self,
                 *,
                 encoder_config,
                 decoder_config,
                 quantizer_config,

                 lossconfig=None,

                 ckpt_path=None,

                 ignore_keys=[],
                 trainable=False,
                 train_part='all',

                 im_process_info={'scale': 127.5, 'mean': 1.0, 'std': 1.0},
                 resize_mask_type='pixel_shuffle',
                 combine_rec_and_gt=False,

                 part_list=[]
                 ):
        super().__init__()

        ##############################################################
        # MODULE SETTING
        ##############################################################
        # basic module
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.quantize = instantiate_from_config(quantizer_config)

        # other module
        self.quant_conv = torch.nn.Conv2d(self.encoder.out_channels, self.quantize.e_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.quantize.e_dim, self.decoder.in_channels, 1)

        # loss setting
        if lossconfig is not None and trainable:
            self.loss = instantiate_from_config(lossconfig)
        else:
            self.loss = None
        ##############################################################
        # PARAMETER SETTING
        ##############################################################
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.trainable = trainable
        self.train_part = train_part
        self._set_trainable(train_part=self.train_part)
        ##############################################################
        # IMAGE PROCESSING SETTING
        ##############################################################
        self.im_process_info = im_process_info
        for k, v in self.im_process_info.items():
            v = torch.tensor(v).view(1, -1, 1, 1)
            if v.shape[1] != 3:
                v = v.repeat(1, 3, 1, 1)
            self.im_process_info[k] = v

        self.combine_rec_and_gt = combine_rec_and_gt
        self.resize_mask_type = resize_mask_type
        ##############################################################

        self.set_part_trainable(part_list)

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
                    print("P-VQVAE: Deleting key {} from state_dict.".format(k))
                    del sd[k]
        keys = list(sd.keys())
        print(keys)
        # for k in keys:
        #     if 'codec' in k:
        #         _k = k[6:]
        #         sd[_k] = sd[k]
        #         del sd[k]
        # keys = list(sd.keys())
        # print(keys)
        self.load_state_dict(sd, strict=False)
        print("P-VQVAE: Load pretrained model from {}".format(path))

    def set_part_trainable(self, part_list):
        if 'decoder' in part_list:
            self.decoder.train()
            for pn, p in self.decoder.named_parameters():
                p.requires_grad = True
            self.post_quant_conv.train()
            for pn, p in self.post_quant_conv.named_parameters():
                p.requires_grad = True
            print('set decoder to trainable')
        if 'encoder' in part_list:
            self.encoder.train()
            for pn, p in self.decoder.named_parameters():
                p.requires_grad = True
            print('set encoder to trainable')

    @property
    def device(self):
        return self.post_quant_conv.weight.device

    @property
    def embed_dim(self):
        return self.post_quant_conv.weight.shape[0]

    def get_codebook(self):
        return self.quantize.get_codebook()

    def pre_process(self, data):
        data = data.to(self.device)
        # data = data / 127.5 - 1.0
        data = (data / self.im_process_info['scale'].to(data.device) - self.im_process_info['mean'].to(data.device)) / self.im_process_info['std'].to(data.device)
        return data

    def multi_pixels_with_mask(self, data, mask):
        if self.im_process_info['mean'].sum() != 0.0:
            eps = 1e-3
            if data.max() > (((255.0 / self.im_process_info['scale'] - self.im_process_info['mean'])/self.im_process_info['std']).max().to(data.device) + eps):
                raise ValueError('The data need to be preprocessed! data max: {}'.format(data.max()))
            mask = mask.to(data.device).repeat(1,3,1,1).type(torch.bool)
            data_m = data * mask.to(data.dtype)
            data_m[~mask] = ((torch.zeros_like(data_m) - self.im_process_info['mean'].to(data_m.device)) / self.im_process_info['std'].to(data_m.device))[~mask]
        else:
            data_m = data * mask.to(data)
        return data_m

    def post_process(self, data):
        # data = (data + 1.0) * 127.5
        data = (data * self.im_process_info['std'].to(data.device) + self.im_process_info['mean'].to(data.device)) * self.im_process_info['scale'].to(data.device)
        data = torch.clamp(data, min=0.0, max=255.0)
        return data

    def get_number_of_tokens(self):
        return self.quantize.n_e

    @torch.no_grad()
    def get_features(self, data, mask=None, 
                     return_token=False, 
                     return_quantize_feature=False):
        """
        Get the feature from image
        
        """
        data = self.pre_process(data)
        if mask is not None:
            data = self.multi_pixels_with_mask(data, mask)
        x = self.encoder(data)
        x = self.quant_conv(x) # B C H W
        token_shape = (x.shape[-2], x.shape[-1])

        output = {
            'feature': F.normalize(x, dim=1, p=2) if self.quantize.norm_feat else x
        }
        if return_quantize_feature or return_token:
            if mask is not None:
                token_type = get_token_type(mask, token_shape, type=self.resize_mask_type) # B x 1 x H x W
            else:
                token_type = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).long().to(self.device)

            quant_out = self.quantize(x, token_type=token_type)
            if return_quantize_feature:
                output['feature_quantize'] = quant_out['quantize']
            if return_token:
                output['token'] = quant_out['index']
        output['token_shape'] = token_shape
        return output
    
    def decode_with_feature(self, feature, masked_image, mask, post_conv=False, pre_process=False, post_process=False):
        if pre_process:
            masked_image = self.pre_process(masked_image)
        if post_conv:
            feature = self.post_quant_conv(feature)
        decode_output = self.decoder(feature, masked_image, mask=mask)
        if post_process:
            decode_output = self.post_process(decode_output)

        return decode_output
    
    @torch.no_grad()
    def get_codebook_entry_with_token(self, token, **kwargs):
        """
        token: B x L

        return:
            feature: features, B x L x C
        """

        t_shape = token.shape
        feat = self.quantize.get_codebook_entry(token.view(-1), shape=t_shape) 
        return {'feature': feat}

    @torch.no_grad()
    def get_tokens_with_feature(self, feat, token_type=None, topk=1):
        if token_type is None:
            token_type = torch.ones((feat.shape[0], 1, feat.shape[2], feat.shape[3])).long().to(self.device)
        idx = self.quantize(feat, token_type=token_type, topk=topk)['index']
        return idx

    @torch.no_grad()
    def get_tokens(self, data, mask=None, erase_mask=None, topk=1, return_token_index=False, cache=True, **kwargs):
        """
        Get the tokens of the given images
        """
        data = self.pre_process(data)
        x = self.encoder(data)
        x = self.quant_conv(x)
        token_shape = (x.shape[-2], x.shape[-1])

        if erase_mask is not None:
            token_type_erase = get_token_type(erase_mask, token_shape, type=self.resize_mask_type) # B x 1 x H x W
        else:
            token_type_erase = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).long().to(self.device)

        idx = self.quantize(x, token_type=token_type_erase, topk=topk)['index']

        # import pdb; pdb.set_trace()
        if cache:
            self.mask_im_tmp = self.multi_pixels_with_mask(data, mask)
            self.mask_tmp = mask

        output = {}
        output['token'] = idx.view(idx.shape[0], -1)

        # import pdb; pdb.set_trace()
        if mask is not None: # mask should be B x 1 x H x W
            # downsampling
            # mask = F.interpolate(mask.float(), size=idx_mask.shape[-2:]).to(torch.bool)
            token_type = get_token_type(mask, token_shape, type=self.resize_mask_type) # B x 1 x H x W
            mask = token_type == 1
            output = {
                'target': idx.view(idx.shape[0], -1).clone(),
                'mask': mask.view(mask.shape[0], -1),
                'token': idx.view(idx.shape[0], -1),
                'token_type': token_type.view(token_type.shape[0], -1),
            }
        else:
            output = {
                'token': idx.view(idx.shape[0], -1)
                }

        # get token index
        # used for computing token frequency
        if return_token_index:
            token_index = output['token'] #.view(-1)
            output['token_index'] = token_index
        output['token_shape'] = token_shape
        return output

    @torch.no_grad()
    def decode(self, token, token_shape, mask_im=None, mask=None, combine_rec_and_gt=True):
        """
        Decode the image with provided tokens
        """
        bhw = (token.shape[0], token_shape[0], token_shape[1])
        quant = self.quantize.get_codebook_entry(token.view(-1), shape=bhw)
        quant = self.post_quant_conv(quant)

        if mask_im is None:
            rec = self.decoder(quant, self.mask_im_tmp, mask=self.mask_tmp)
        else:
            rec = self.decoder(quant, self.pre_process(mask_im), mask=mask)

        if combine_rec_and_gt and self.combine_rec_and_gt:
            if mask_im is None:
                rec = rec * (1-self.mask_tmp.to(rec.dtype)) + self.mask_im_tmp * self.mask_tmp.to(rec.dtype)
            else:
                rec = rec * (1-mask.to(rec.dtype)) + self.pre_process(mask_im) * mask.to(rec.dtype)
        rec = self.post_process(rec)
        return rec

    @torch.no_grad()
    def sample(self, batch):

        data = self.pre_process(batch['image'])
        x = self.encoder(data)
        x = self.quant_conv(x)
        token_shape = (x.shape[-2], x.shape[-1])

        if 'erase_mask' in batch:
            token_type_erase = get_token_type(batch['erase_mask'], token_shape, type=self.resize_mask_type).to(self.device) # B x 1 x H x W
        else:
            token_type_erase = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).long().to(self.device)

        quant = self.quantize(x, token_type=token_type_erase)['quantize']
        quant = self.post_quant_conv(quant)
        mask_im = self.multi_pixels_with_mask(data, batch['mask'])
        rec = self.decoder(quant, mask_im, mask=batch['mask'])

        rec = self.post_process(rec)

        out = {'input': batch['image'], 'reconstruction': rec}
        out['reference_input'] = self.post_process(mask_im)
        out['reference_mask'] = batch['mask'] * 255
        return out

    def get_last_layer(self):
        return self.decoder.post_layers[-1].weight
    
    def parameters(self, recurse=True, name=None):
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            if name == 'generator':
                params = list(self.encoder.parameters())+ \
                         list(self.decoder.parameters())+\
                         list(self.quantize.parameters())+\
                         list(self.quant_conv.parameters())+\
                         list(self.post_quant_conv.parameters())
            elif name == 'discriminator':
                params = self.loss.discriminator.parameters()
            elif name == 'decoder_tuning':
                params = list(self.decoder.parameters())+\
                         list(self.post_quant_conv.parameters())
            else:
                raise ValueError("Unknown type of name {}".format(name))
            return params

    def forward(self, batch, name='none', step=0, total_steps=None):
        
        if name == 'generator':
            input = self.pre_process(batch['image'])
            x = self.encoder(input)
            x = self.quant_conv(x)
            token_shape = list(x.shape[-2:])

            if 'erase_mask' in batch:
                token_type_erase = get_token_type(batch['erase_mask'], token_shape, type=self.resize_mask_type) # B x 1 x H x W
            else:
                token_type_erase = torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).long().to(self.device)

            quant_out = self.quantize(x, token_type=token_type_erase, step=step, total_steps=total_steps)
            quant = quant_out['quantize']

            # recconstruction
            quant = self.post_quant_conv(quant)
            rec = self.decoder(quant, self.multi_pixels_with_mask(input, batch['mask']), mask=batch['mask'])

            input = self.post_process(input)
            rec = self.post_process(rec)

            return input, rec, quant_out

        elif name == 'discriminator':
            if self.loss.norm_to_0_1:
                loss_im = self.post_process(self.input_tmp) / 255.0
                loss_rec = self.post_process(self.rec_tmp) / 255.0
            else:
                loss_im = self.input_tmp
                loss_rec = self.rec_tmp
            loss_im = self.input_tmp
            loss_rec = self.rec_tmp
            output = self.loss(
                image=loss_im, # norm to [0, 1]
                reconstruction=loss_rec, # norm to [0, 1]
                step=step,
                name=name
            )
        else:
            raise NotImplementedError('{}'.format(name))        
        return output
