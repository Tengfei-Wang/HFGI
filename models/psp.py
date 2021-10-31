import matplotlib

matplotlib.use('Agg')
import torch
from torch import nn
from models.encoders import psp_encoders
from models.stylegan2.model import Generator #, Discriminator
from configs.paths_config import model_paths
import torchvision.transforms as transforms


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # Define architecture
        self.encoder = self.set_encoder()
        self.residue =  psp_encoders.ResidualEncoder() #Ec
        self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.grid_transform = transforms.RandomPerspective(distortion_scale=opts.distortion_scale, p=opts.aug_rate)
        self.grid_align = psp_encoders.ResidualAligner() #ADA
        #self.discriminator = Discriminator(1024)
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading basic encoder from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)

            if not self.opts.is_train:
                self.residue.load_state_dict(get_keys(ckpt, 'residue'), strict=True)
                self.grid_align.load_state_dict(get_keys(ckpt, 'grid_align'), strict=True)
       
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes], None,
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

    
        imgs_ = torch.nn.functional.interpolate(torch.clamp(images, -1., 1.), size=(256,256) , mode='bilinear') 
        res_gt = (x - imgs_ ).detach() 
        res_unaligned = self.grid_transform(res_gt).detach() 
    

        res_aligned = self.grid_align(torch.cat((res_unaligned, imgs_ ), 1))
        res = res_aligned.to(self.opts.device)
 
        delta = res - res_gt
        conditions = self.residue(res)
        if conditions is not None:
            images, result_latent = self.decoder([codes], conditions,
                                                input_is_latent=input_is_latent,
                                                randomize_noise=randomize_noise,
                                                return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent, delta, imgs_
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
