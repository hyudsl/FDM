import torch
import torch.nn as nn

from image_synthesis.utils.misc import instantiate_from_config
from image_synthesis.modeling.utils.misc import get_token_type

from image_synthesis.modeling.utils.patch_processing import patches2img, img2patches

import numpy as np

class InpaintingModel(nn.Module):
    def __init__(self,
                 content_codec_config,
                 fdm_config=None,
                 generator_config=None,

                 codec_path=None,
                 generator_path=None,
                 ignore_keys=[],

                 mode="none",
                 loss_config=None):
        super().__init__()
        self.content_codec = instantiate_from_config(content_codec_config)

        if fdm_config != None:
            self.fdm = instantiate_from_config(fdm_config)
        else:
            self.fdm = None

        if generator_config != None:
            self.generator = instantiate_from_config(generator_config)
        else:
            self.generator = None

        self.temp = nn.Parameter(torch.randn(1), requires_grad=False)

        self.stride = self.content_codec.encoder.stride
    
        if codec_path != None:
            self.init_from_ckpt(codec_path, generator_path, ignore_keys)

        self.mode = mode

        if loss_config is not None:
            self.loss = instantiate_from_config(loss_config)
        else:
            self.loss = None

    @property
    def device(self):
        return self.temp.device

    def init_from_ckpt(self, path, generator_path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        if 'model' in sd:
            sd = sd['model']
        else:
            sd = sd["state_dict"]

        keys = list(sd.keys())
        for k in keys:
            if 'lip' in k:
                _k = k[4:]
                _k = 'fdm.'+_k
                sd[_k] = sd[k]
                del sd[k]
                k=_k

        if generator_path != None:
            generator_sd = torch.load(generator_path, map_location="cpu")
            if 'model' in generator_sd:
                generator_sd = generator_sd['model']
            else:
                generator_sd = generator_sd["state_dict"]
            generator_keys = list(generator_sd.keys())
            for k in generator_keys:
                if k.startswith('generator'):
                    sd[k] = generator_sd[k]

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("InpaintingModel: Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print("InpaintingModel: Load pretrained model from {}".format(path))

    def parameters(self, recurse=True, name=None):
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            if name == 'fdm':
                params = list(self.fdm.parameters())
            elif name == 'fdm_decoder_tuning':
                params = list(self.content_codec.parameters(recurse, 'decoder_tuning'))+ \
                         list(self.fdm.parameters())
            elif name == 'generator':
                params = list(self.content_codec.parameters(recurse, name))
            elif name == 'discriminator':
                params = list(self.loss.discriminator.parameters())
            elif name == 'transformer':
                params = list(self.generator.parameters(recurse, name))
            else:
                raise ValueError("Unknown type of name {}".format(name))
            return params

    @torch.no_grad()
    def get_masked_patch_index(self, mask):
        """
        mask: B x 1 x H x W, 0: masked, 1: unmasked
 
        return:
            mask_patches: B x 1 x h x w, 0: unmasked 1: masked
        """
        h, w = mask.shape[-2], mask.shape[-1]
        assert h==w, 'Height and width must be te same, {}!={}'.format(h,w)
        
        patch_num = h//self.stride

        mask_patches = img2patches(mask.to(torch.float32), self.stride, patch_num)
        B, C, H, W = mask_patches.shape
        mask_patches = mask_patches.reshape(B,C*H*W)
        mask_patches = torch.mean(mask_patches, axis=1)
        mask_patches.reshape(B, 1)
        mask_patches = patches2img(mask_patches, patch_num, 1, patch_num)
        mask_patches = mask_patches < 1
        return mask_patches.to(torch.float32)

    def apply_fdm(self, quantized_vector, background_vector, mask, return_pv=False):
        """
        quantized_vector: B x C x H x W
        background_vector: B x C x H x W
        mask: B x 1 x H x W
 
        return:
            v: retrieved vector, B x C x H x W
            pv: predicted vector,  B x C x H x W
        """

        masked_patch_idx = self.get_masked_patch_index(mask)

        qv = background_vector*(1-masked_patch_idx) + quantized_vector*masked_patch_idx
        pv = self.fdm(qv, masked_patch_idx)
        v = qv+pv

        if return_pv:
            return v, pv

        return v

    def content_generate(self, feature, token, token_type, generate_config=None):
        if generate_config is None:
            filter_ratio = 1
            num_token_per_iter = 1
        else:
            filter_ratio = generate_config.filter_ratio
            num_token_per_iter = generate_config.num_token_per_iter
        
        b, _, h, w = feature.shape
        feat_mask = feature.permute(0, 2, 3, 1).view(b, h*w, -1).contiguous()
    
        content_feat = feat_mask # B x HW x D
        content_token = token.view(b, -1) # B x HW
        original_mask = (token_type == 1).view(b, 1, h, w) # B x HW
        content_mask = original_mask.clone().view(b, -1)    # True : unmasked, False : masked
        original_mask = original_mask.to(torch.float32)

        while content_mask.sum() < content_mask.numel():
            # print(content_mask.sum(), "/", content_mask.numel())
            batch_idx = content_mask.sum(dim=-1) < content_mask.shape[-1] # B

            emb = content_feat[batch_idx] # b x HW x D
            mask_ = content_mask[batch_idx]

            sample, breaked, other_out = self.generator.get_sample_token(emb, mask_, filter_ratio=filter_ratio, num_token_per_iter=num_token_per_iter,pos_embedding=True)

            if breaked:
                content_token_tmp = content_token[batch_idx] # b x HW
                content_mask_tmp = content_mask[batch_idx] # b x HW
                sample_tmp = sample[batch_idx]
                content_token_tmp = content_token_tmp * content_mask_tmp + sample_tmp * (~content_mask_tmp)
                content_token[batch_idx] = content_token_tmp
                content_feat[batch_idx] = self.content_codec.get_codebook_entry_with_token(content_token[batch_idx])['feature']
                break

            index = other_out['index']

            # change contents
            sample_feat = self.content_codec.get_codebook_entry_with_token(sample)['feature'] # b x num x C

            content_feat_tmp = content_feat[batch_idx] # b x HW x D
            content_token_tmp = content_token[batch_idx] # b x HW
            content_mask_tmp = content_mask[batch_idx] # b x HW
            for i in range(sample_feat.shape[0]):
                for j in range(index.shape[1]):
                    if not content_mask_tmp[i][index[i,j]]:
                        content_feat_tmp[i][index[i,j]] = sample_feat[i, j]
                        content_token_tmp[i][index[i,j]] = sample[i, j]
                        content_mask_tmp[i][index[i,j]] = True

            content_feat[batch_idx] = content_feat_tmp
            content_token[batch_idx] = content_token_tmp 
            content_mask[batch_idx] = content_mask_tmp

        generated_feature = content_feat.permute(0,2,1).view(b, -1, h, w)

        out = {'generated_token': content_token}

        out['generated_feature'] = generated_feature

        return out

    def sample(self, batch, generate_config=None, save_token=False):
        image = batch['image'].cuda()
        mask = batch['mask'].cuda()

        org_data = self.content_codec.get_features(image,
                                            return_quantize_feature=True) # B x C x H x W
        org_quant_feature = org_data['feature_quantize']
        generated_feature = org_quant_feature

        org_data = self.content_codec.get_features(image,
                                            return_quantize_feature=True) # B x C x H x W
        org_quant_feature = org_data['feature_quantize']

        input_data = self.content_codec.get_features(image, 
                                            mask=mask, 
                                            return_quantize_feature=False,
                                            return_token=True) # B x C x H x W
        input_feature = input_data['feature']
        input_token = input_data['token']

        b, _, h, w = input_feature.shape
        token_type = get_token_type(mask, type='pixel_shuffle', token_shape=[h,w])

        if 'token' in batch.keys():
            generated_feature = self.content_codec.get_codebook_entry_with_token(batch['token'].view(b,-1))['feature'].permute(0,2,1).view(b, -1, h, w)
        elif self.generator != None and 'token' not in batch.keys():
            out = self.content_generate(input_feature, input_token, token_type, generate_config)
            generated_feature = out['generated_feature']
            generated_token = out['generated_token']

        if self.fdm != None:
            generated_feature, pv = self.apply_fdm(generated_feature, input_feature, mask, return_pv=True)

        inpainted_image = self.content_codec.decode_with_feature(generated_feature, image*mask, mask, post_conv=True, pre_process=True, post_process=True)

        out = {'input': image, 'masked_input': image*mask, 'inpainted_image': inpainted_image}

        if save_token and self.generator != None:
            out['token'] = generated_token

        return out

        # return input_data
    
    def forward(self, batch, epoch=0, name='none', step=0, total_steps=None, return_loss=True):
        if name == 'fdm':
            image = batch['image']
            mask = batch['mask']

            masked_patch_idx = self.get_masked_patch_index(mask)

            input_data = self.content_codec.get_features(image,
                                                return_quantize_feature=True) # B x C x H x W
            background_feature = input_data['feature']
            quantized_feature = input_data['feature_quantize']

            _, lost_feature = self.apply_fdm(quantized_feature, background_feature, mask, True)

            gt_feature = (background_feature-quantized_feature)*masked_patch_idx

            output = self.loss(
                image=gt_feature, # norm to [0, 1]
                reconstruction=lost_feature, # norm to [0, 1]
                mask=mask,
                step=step,
                name=name,
                other_input={}
            )

            return output
        elif name == 'fdm_decoder_tuning':
            image = batch['image']
            mask = batch['mask']

            masked_patch_idx = self.get_masked_patch_index(mask)

            input_data = self.content_codec.get_features(image,
                                                return_quantize_feature=True) # B x C x H x W
            background_feature = input_data['feature']
            quantized_feature = input_data['feature_quantize']

            repaired_feature, lost_feature = self.apply_fdm(quantized_feature, background_feature, mask, True)

            gt_feature = (background_feature-quantized_feature)*masked_patch_idx

            rec = self.content_codec.decode_with_feature(repaired_feature, image*mask, mask, post_conv=True, pre_process=True, post_process=True)

            self.input_tmp = batch['image']
            self.rec_tmp = rec
            self.mask_tmp = mask

            loss_im = batch['image'] / 255.0
            loss_rec = rec / 255.0

            output = self.loss(
                image=loss_im, # norm to [0, 1]
                reconstruction=loss_rec, # norm to [0, 1]
                mask=mask,
                step=step,
                name=name,
                other_input={'lost_feature': lost_feature, 'gt_feature': gt_feature.detach()}
            )
        elif name == 'generator':
            input, rec, quant_out = self.content_codec(batch, name, step, total_steps)

            # save some tensors for 
            self.input_tmp = input 
            self.rec_tmp = rec 
            self.mask_tmp = batch['mask']

            other_loss = {}
            for k in quant_out:
                if 'loss' in k:
                    other_loss[k] = quant_out[k]
            
            # norm image to 0-1
            loss_im = self.input_tmp / 255.0
            loss_rec = self.rec_tmp / 255.0
            
            output = self.loss(
                image=loss_im, # norm to [0, 1]
                reconstruction=loss_rec, # norm to [0, 1]
                mask=batch['mask'],
                step=step,
                name=name,
                other_loss=other_loss
            )
            
            # for observing the number of used codebooks
            for k, v in quant_out.items():
                if k == 'loss':
                    continue
                if v.numel() == 1 and len(v.shape) == 0:
                    output[k] = v
        elif name == 'discriminator':
            loss_im = self.input_tmp / 255.0
            loss_rec = self.rec_tmp / 255.0

            output = self.loss(
                image=loss_im, # norm to [0, 1]
                reconstruction=loss_rec, # norm to [0, 1]
                mask=self.mask_tmp,
                step=step,
                name=name
            )
        elif name == 'transformer':
            data_mask =  self.content_codec.get_features(
                batch['image'], 
                mask=batch['mask'], 
                return_quantize_feature=True,
                return_token=False
                )
            gt_features = self.content_codec.get_features(batch['image'], return_token=True)
            data_mask['gt_features'] = gt_features
            output = self.generator(batch, data_mask, return_loss=True, return_logits=False)

        return output
