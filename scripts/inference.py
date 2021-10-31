import argparse
import torch
import numpy as np
import sys
import os

sys.path.append(".")
sys.path.append("..")

from configs import data_configs, paths_config
from datasets.inference_dataset import InferenceDataset
from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from PIL import Image
from editings import latent_editor

def main(args):
    net, opts = setup_model(args.ckpt, device)
    is_cars = 'car' in opts.dataset_type
    generator = net.decoder
    generator.eval()
    aligner = net.grid_align
    args, data_loader = setup_data_loader(args, opts)
    editor = latent_editor.LatentEditor(net.decoder, is_cars)

    # initial inversion
    latent_codes = get_all_latents(net, data_loader, args.n_sample, is_cars=is_cars)

    # set the editing operation
    if args.edit_attribute == 'inversion':
        pass
    elif args.edit_attribute == 'age' or args.edit_attribute == 'smile':
        interfacegan_directions = {
                'age': './editings/interfacegan_directions/age.pt',
                'smile': './editings/interfacegan_directions/smile.pt' }
        edit_direction = torch.load(interfacegan_directions[args.edit_attribute]).to(device)
    else:
        ganspace_pca = torch.load('./editings/ganspace_pca/ffhq_pca.pt') 
        ganspace_directions = {
            'eyes':            (54,  7,  8,  20),
            'beard':           (58,  7,  9,  -20),
            'lip':             (34, 10, 11,  20) }            
        edit_direction = ganspace_directions[args.edit_attribute]

    edit_directory_path = os.path.join(args.save_dir, args.edit_attribute)
    os.makedirs(edit_directory_path, exist_ok=True)

    # perform high-fidelity inversion or editing
    for i, batch in enumerate(data_loader):
        if args.n_sample is not None and i > args.n_sample:
            print('inference finished!')
            break            
        x = batch.to(device).float()

        # calculate the distortion map
        imgs, _ = generator([latent_codes[i].unsqueeze(0).to(device)],None, input_is_latent=True, randomize_noise=False, return_latents=True)
        res = x -  torch.nn.functional.interpolate(torch.clamp(imgs, -1., 1.), size=(256,256) , mode='bilinear')

        # produce initial editing image
        # edit_latents = editor.apply_interfacegan(latent_codes[i].to(device), interfacegan_direction, factor_range=np.linspace(-3, 3, num=40))  
        if args.edit_attribute == 'inversion':
            img_edit = imgs
            edit_latents = latent_codes[i].unsqueeze(0).to(device)
        elif args.edit_attribute == 'age' or args.edit_attribute == 'smile':
            img_edit, edit_latents = editor.apply_interfacegan(latent_codes[i].unsqueeze(0).to(device), edit_direction, factor=args.edit_degree)
        else:
            img_edit, edit_latents = editor.apply_ganspace(latent_codes[i].unsqueeze(0).to(device), ganspace_pca, [edit_direction])

        # align the distortion map
        img_edit = torch.nn.functional.interpolate(torch.clamp(img_edit, -1., 1.), size=(256,256) , mode='bilinear')
        res_align  = net.grid_align(torch.cat((res, img_edit  ), 1))

        # consultation fusion
        conditions = net.residue(res_align)
        imgs, _ = generator([edit_latents],conditions, input_is_latent=True, randomize_noise=False, return_latents=True)
        if is_cars:
            imgs = imgs[:, :, 64:448, :]
            
        # save images
        imgs = torch.nn.functional.interpolate(imgs, size=(256,256) , mode='bilinear')
        result = tensor2im(imgs[0])
        im_save_path = os.path.join(edit_directory_path, f"{i:05d}.jpg")
        Image.fromarray(np.array(result)).save(im_save_path)

def setup_data_loader(args, opts):
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    images_path = args.images_dir if args.images_dir is not None else dataset_args['test_source_root']
    print(f"images path: {images_path}")
    align_function = None
    test_dataset = InferenceDataset(root=images_path,
                                    transform=transforms_dict['transform_test'],
                                    preprocess=align_function,
                                    opts=opts)

    data_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             num_workers=2,
                             drop_last=True)

    print(f'dataset length: {len(test_dataset)}')

    if args.n_sample is None:
        args.n_sample = len(test_dataset)
    return args, data_loader


def get_latents(net, x, is_cars=False):
    codes = net.encoder(x)
    if net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + net.latent_avg.repeat(codes.shape[0], 1, 1)
    if codes.shape[1] == 18 and is_cars:
        codes = codes[:, :16, :]
    return codes


def get_all_latents(net, data_loader, n_images=None, is_cars=False):
    all_latents = []
    i = 0
    with torch.no_grad():
        for batch in data_loader:
            if n_images is not None and i > n_images:
                break
            x = batch
            inputs = x.to(device).float()
            latents = get_latents(net, inputs, is_cars)
            all_latents.append(latents)
            i += len(latents)
    return torch.cat(all_latents)


def save_image(img, save_dir, idx):
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, f"{idx:05d}.jpg")
    Image.fromarray(np.array(result)).save(im_save_path)

if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--images_dir", type=str, default=None, help="The directory to the images")
    parser.add_argument("--save_dir", type=str, default=None, help="The directory to save.")
    parser.add_argument("--batch", type=int, default=1, help="batch size for the generator")
    parser.add_argument("--n_sample", type=int, default=None, help="number of the samples to infer.")
    parser.add_argument("--edit_attribute", type=str, default='smile', help="The desired attribute")
    parser.add_argument("--edit_degree", type=float, default=0, help="edit degreee")
    parser.add_argument("ckpt", metavar="CHECKPOINT", help="path to generator checkpoint")

    args = parser.parse_args()
    main(args)