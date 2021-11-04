import tempfile
import numpy as np
from argparse import Namespace
from pathlib import Path
import torch
from torchvision import transforms
import PIL.Image
import scipy
import scipy.ndimage
import dlib
import imageio
import cog
from models.psp import pSp
from utils.common import tensor2im
from editings import latent_editor

"""
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
"""


class Predictor(cog.Predictor):
    def setup(self):
        model_path = "checkpoint/ckpt.pt"
        ckpt = torch.load(model_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["is_train"] = False
        opts["checkpoint_path"] = model_path
        opts = Namespace(**opts)
        self.net = pSp(opts)
        self.net.eval()
        self.net.cuda()
        print("Model successfully loaded!")
        self.img_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.editor = latent_editor.LatentEditor(self.net.decoder)
        # interface-GAN
        interfacegan_directions = {
            "age": "./editings/interfacegan_directions/age.pt",
            "smile": "./editings/interfacegan_directions/smile.pt",
        }
        self.ganspace_pca = torch.load("./editings/ganspace_pca/ffhq_pca.pt")
        self.edit_direction = {
            "age": torch.load(interfacegan_directions["age"]).cuda(),
            "smile": torch.load(interfacegan_directions["smile"]).cuda(),
            "eyes": (54, 7, 8, 20),
            "beard": (58, 7, 9, -20),
            "lip": (34, 10, 11, 20),
        }

    @cog.input(
        "image",
        type=Path,
        help="input facial image, which will be aligned and cropped to 256*256 first",
    )
    @cog.input(
        "edit_attribute",
        type=str,
        default="smile",
        options=["inversion", "age", "smile", "eyes", "lip", "beard"],
        help="choose image editing option",
    )
    @cog.input(
        "edit_degree",
        type=float,
        default=0,
        min=-5,
        max=5,
        help="control the degree of editing (valid for 'age' and 'smile').",
    )
    def predict(self, image, edit_attribute, edit_degree):
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        resize_dims = (256, 256)
        input_path = str(image)
        # for replicate, webcam input might be rgba, convert to rgb first
        input = imageio.imread(input_path)
        if input.shape[-1] == 4:
            rgba_image = PIL.Image.open(input_path)
            rgb_image = rgba_image.convert("RGB")
            input_path = "rgb_input.png"
            imageio.imwrite(input_path, rgb_image)

        # align and crop image
        input_image = run_alignment(input_path)
        input_image.resize(resize_dims)
        transformed_image = self.img_transforms(input_image)
        x = transformed_image.unsqueeze(0).cuda()
        latent_codes = get_latents(self.net, x)

        # calculate the distortion map
        imgs, _ = self.net.decoder(
            [latent_codes[0].unsqueeze(0).cuda()],
            None,
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True,
        )
        res = x - torch.nn.functional.interpolate(
            torch.clamp(imgs, -1.0, 1.0), size=(256, 256), mode="bilinear"
        )

        # ADA
        img_edit = torch.nn.functional.interpolate(
            torch.clamp(imgs, -1.0, 1.0), size=(256, 256), mode="bilinear"
        )
        res_align = self.net.grid_align(torch.cat((res, img_edit), 1))

        # consultation fusion
        conditions = self.net.residue(res_align)

        if edit_attribute == "inversion":
            result, _ = self.net.decoder(
                [latent_codes],
                conditions,
                input_is_latent=True,
                randomize_noise=False,
                return_latents=True,
            )
        else:
            edit_direction = self.edit_direction[edit_attribute]
            if edit_attribute in ["age", "smile"]:
                img_edit, edit_latents = self.editor.apply_interfacegan(
                    latent_codes[0].unsqueeze(0).cuda(),
                    edit_direction,
                    factor=edit_degree,
                )

            else:
                img_edit, edit_latents = self.editor.apply_ganspace(
                    latent_codes[0].unsqueeze(0).cuda(),
                    self.ganspace_pca,
                    [edit_direction],
                )

            result, _ = self.net.decoder(
                [edit_latents],
                conditions,
                input_is_latent=True,
                randomize_noise=False,
                return_latents=True,
            )
            result = torch.nn.functional.interpolate(
                result, size=(256, 256), mode="bilinear"
            )

        result = tensor2im(result[0])
        PIL.Image.fromarray(np.array(result)).save(str(out_path))
        PIL.Image.fromarray(np.array(result)).save("ooo.png")
        return out_path


def get_landmark(filepath, predictor):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()

    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    return lm


def align_face(filepath, predictor):
    """
    :param filepath: str
    :return: PIL Image
    """

    lm = get_landmark(filepath, predictor)

    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    img = PIL.Image.open(filepath)

    output_size = 256
    transform_size = 256
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect"
        )
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(
            mask * 3.0 + 1.0, 0.0, 1.0
        )
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), "RGB")
        quad += pad[:2]

    # Transform.
    img = img.transform(
        (transform_size, transform_size),
        PIL.Image.QUAD,
        (quad + 0.5).flatten(),
        PIL.Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Return aligned image.
    return img


def run_alignment(image_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


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
