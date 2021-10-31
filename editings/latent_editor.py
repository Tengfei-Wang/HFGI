import torch
import sys
sys.path.append(".")
sys.path.append("..")
from editings import ganspace

class LatentEditor(object):
    def __init__(self, stylegan_generator, is_cars=False):
        self.generator = stylegan_generator
        self.is_cars = is_cars  # Since the cars StyleGAN output is 384x512, there is a need to crop the 512x512 output.

    def apply_ganspace(self, latent, ganspace_pca, edit_directions):
        edit_latents = ganspace.edit(latent, ganspace_pca, edit_directions)
        return self._latents_to_image(edit_latents), edit_latents

    def apply_interfacegan(self, latent, direction, factor=None):
        edit_latents = latent + factor * direction
        return self._latents_to_image(edit_latents), edit_latents

    def _latents_to_image(self, latents):
        with torch.no_grad():
            images, _ = self.generator([latents], None, randomize_noise=False, input_is_latent=True)
            if self.is_cars:
                images = images[:, :, 64:448, :]  # 512x512 -> 384x512
        return images
