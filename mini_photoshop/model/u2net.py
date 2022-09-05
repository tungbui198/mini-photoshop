import math
import os
from collections import namedtuple
from os.path import abspath, dirname

import cv2
import gdown
import numpy as np
import torch
import torch.nn as nn

from mini_photoshop.utils import norm_img, resize_max_size

ModelConfig = namedtuple("ModelConfig", ["name", "url"])


u2net_lite = ModelConfig(
    name="u2net_lite",
    url="https://drive.google.com/uc?id=1iACXN1N2AqvbdojnUwsQ9sz6BZj2zEXn",
)
u2net = ModelConfig(
    name="u2net",
    url="https://drive.google.com/uc?id=1st1TfjUn7wN0seWhWidmS80AYtNsCsyc",
)
u2net_human_seg = ModelConfig(
    name="u2net_human_seg",
    url="https://drive.google.com/uc?id=1fx5J0wV3CK5eNNzbU30P4fUXvKYbTrZH",
)

u2net_list = [u2net_lite, u2net, u2net_human_seg]


def _upsample_like(x, size):
    return nn.Upsample(size=size, mode="bilinear", align_corners=False)(x)


def _size_map(x, height):
    # {height: size} for Upsample
    size = list(x.shape[-2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(
            in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate
        )
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False):
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f"rebnconv{height}")(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, "downsample")(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f"rebnconv{height}d")(torch.cat((x2, x1), 1))
                return (
                    _upsample_like(x, sizes[height - 1])
                    if not self.dilated and height > 1
                    else x
                )
            else:
                return getattr(self, f"rebnconv{height}")(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
        self.add_module("rebnconvin", REBNCONV(in_ch, out_ch))
        self.add_module(
            "downsample", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        self.add_module("rebnconv1", REBNCONV(out_ch, mid_ch))
        self.add_module("rebnconv1d", REBNCONV(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(
                f"rebnconv{i}", REBNCONV(mid_ch, mid_ch, dilate=dilate)
            )
            self.add_module(
                f"rebnconv{i}d", REBNCONV(mid_ch * 2, mid_ch, dilate=dilate)
            )

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(
            f"rebnconv{height}", REBNCONV(mid_ch, mid_ch, dilate=dilate)
        )


class U2NET(nn.Module):
    def __init__(self, cfgs, out_ch):
        super(U2NET, self).__init__()
        self.out_ch = out_ch
        self._make_layers(cfgs)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        maps = []  # storage for maps

        # side saliency map
        def unet(x, height=1):
            if height < 6:
                x1 = getattr(self, f"stage{height}")(x)
                x2 = unet(getattr(self, "downsample")(x1), height + 1)
                x = getattr(self, f"stage{height}d")(torch.cat((x2, x1), 1))
                side(x, height)
                return (
                    _upsample_like(x, sizes[height - 1]) if height > 1 else x
                )
            else:
                x = getattr(self, f"stage{height}")(x)
                side(x, height)
                return _upsample_like(x, sizes[height - 1])

        def side(x, h):
            # side output saliency map (before sigmoid)
            x = getattr(self, f"side{h}")(x)
            x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # fuse saliency probability maps
            maps.reverse()
            x = torch.cat(maps, 1)
            x = getattr(self, "outconv")(x)
            maps.insert(0, x)
            return [torch.sigmoid(x) for x in maps]

        unet(x)
        maps = fuse()
        return maps

    def _make_layers(self, cfgs):
        self.height = int((len(cfgs) + 1) / 2)
        self.add_module(
            "downsample", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        for k, v in cfgs.items():
            # build rsu block
            self.add_module(k, RSU(v[0], *v[1]))
            if v[2] > 0:
                # build side layer
                self.add_module(
                    f"side{v[0][-1]}",
                    nn.Conv2d(v[2], self.out_ch, 3, padding=1),
                )
        # build fuse layer
        self.add_module(
            "outconv",
            nn.Conv2d(int(self.height * self.out_ch), self.out_ch, 1),
        )


class U2Net:
    def __init__(self, model_name, device):
        """Init class

        Args:
            model_name (str): Model name
            device (str): device
        """
        self.model_name = model_name
        self.device = device
        self.init_model()

    def init_model(self):
        """Init model"""
        model_cfg = None
        for u2net_cfg in u2net_list:
            if u2net_cfg.name == self.model_name:
                model_cfg = u2net_cfg
                break

        if not model_cfg:
            model_cfg = u2net_lite

        parent_dir = dirname(dirname(abspath(__file__)))
        model_path = f"{parent_dir}/pretrained/{model_cfg.name}.pth"

        if not os.path.exists(model_path):
            os.makedirs(dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as model_file:
                gdown.download(url=model_cfg.url, output=model_file)

        if "_lite" in model_cfg.name:
            layer_cfgs = {
                # cfgs for building RSUs and sides
                # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
                "stage1": ["En_1", (7, 3, 16, 64), -1],
                "stage2": ["En_2", (6, 64, 16, 64), -1],
                "stage3": ["En_3", (5, 64, 16, 64), -1],
                "stage4": ["En_4", (4, 64, 16, 64), -1],
                "stage5": ["En_5", (4, 64, 16, 64, True), -1],
                "stage6": ["En_6", (4, 64, 16, 64, True), 64],
                "stage5d": ["De_5", (4, 128, 16, 64, True), 64],
                "stage4d": ["De_4", (4, 128, 16, 64), 64],
                "stage3d": ["De_3", (5, 128, 16, 64), 64],
                "stage2d": ["De_2", (6, 128, 16, 64), 64],
                "stage1d": ["De_1", (7, 128, 16, 64), 64],
            }
        else:
            layer_cfgs = {
                # cfgs for building RSUs and sides
                # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
                "stage1": ["En_1", (7, 3, 32, 64), -1],
                "stage2": ["En_2", (6, 64, 32, 128), -1],
                "stage3": ["En_3", (5, 128, 64, 256), -1],
                "stage4": ["En_4", (4, 256, 128, 512), -1],
                "stage5": ["En_5", (4, 512, 256, 512, True), -1],
                "stage6": ["En_6", (4, 512, 256, 512, True), 512],
                "stage5d": ["De_5", (4, 1024, 256, 512, True), 512],
                "stage4d": ["De_4", (4, 1024, 128, 256), 256],
                "stage3d": ["De_3", (5, 512, 64, 128), 128],
                "stage2d": ["De_2", (6, 256, 32, 64), 64],
                "stage1d": ["De_1", (7, 128, 16, 64), 64],
            }

        model = U2NET(cfgs=layer_cfgs, out_ch=1)
        model.load_state_dict(
            state_dict=torch.load(model_path, map_location="cpu")
        )
        model = model.to(self.device)
        model.eval()
        self.model = model

    def norm_pred(self, d):
        """Normalize prediction

        Args:
            d (torch.Tensor): Decoder Output (prediction)

        Returns:
            torch.Tensor: Normalized prediction
        """
        max = torch.max(d)
        min = torch.min(d)
        dn = (d - min) / (max - min)
        return dn

    def forward(self, image):
        """Forward image to model

        Args:
            image (np.ndarray): Image

        Returns:
            np.ndarray: Mask
        """
        image = norm_img(image)
        image = image = torch.from_numpy(image).unsqueeze(0).to(self.device)

        d1, d2, d3, d4, d5, d6, d7 = self.model(image)
        mask = self.norm_pred(d1[:, 0, :, :])

        mask = mask.squeeze().cpu().data.numpy()
        return mask

    @torch.no_grad()
    def __call__(self, image, resize_limit=512):
        """
        Args:
            image (np.ndarray): Image
            resize_limit (int, optional): max size. Defaults to 512.

        Returns:
            np.ndarray: Mask
        """
        if resize_limit and max(image.shape) > resize_limit:
            origin_size = image.shape[:2]
            downsize_image = resize_max_size(image, size_limit=resize_limit)
            mask = self.forward(downsize_image)
            mask = cv2.resize(
                mask,
                dsize=(origin_size[1], origin_size[0]),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            mask = self.forward(image)
        mask = np.rint(mask).astype(np.uint8)
        return mask
