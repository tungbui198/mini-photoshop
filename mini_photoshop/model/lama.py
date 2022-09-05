import os
from os.path import abspath, dirname
from typing import Optional

import cv2
import gdown
import numpy as np
import torch
from skimage.measure import regionprops

from mini_photoshop.utils import norm_img, pad_img_to_modulo, resize_max_size

LAMA_MODEL_URL = (
    "https://drive.google.com/uc?id=18boxtgk5N69v3eltQMkSVz55a85TomD0"
)
LAMA_MODEL_LOCAL = os.path.join(
    dirname(dirname(abspath(__file__))), "pretrained/big-lama.pt"
)


class LaMa:
    """LaMa Model"""

    pad_mod = 8
    pad_to_square = False
    min_size: Optional[int] = None

    def __init__(self, device):
        """Init class

        Args:
            device (str): device
        """
        self.device = device
        self.init_model()

    def init_model(self):
        """Init model"""
        if not os.path.exists(LAMA_MODEL_LOCAL):
            os.makedirs(dirname(LAMA_MODEL_LOCAL), exist_ok=True)
            with open(LAMA_MODEL_LOCAL, "wb") as model_file:
                gdown.download(url=LAMA_MODEL_URL, output=model_file)
        model_path = LAMA_MODEL_LOCAL
        model = torch.jit.load(model_path, map_location="cpu")
        model = model.to(self.device)
        model.eval()
        self.model = model

    def forward(self, image, mask):
        """Forward model

        Args:
            image (np.ndarray): Image (RGB)
            mask (np.ndarray): Mask

        Returns:
            np.ndarray: Inpainted image (BGR)
        """
        image = norm_img(image)
        mask = norm_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res

    def _pad_forward(self, image, mask):
        """Padding image and mask, then forward model

        Args:
            image (np.ndarray): Image
            mask (np.ndarray): Mask

        Returns:
            np.ndarray: Inpainted Image
        """
        origin_height, origin_width = image.shape[:2]

        regions = regionprops(mask)
        for prop in regions:
            y1, x1, y2, x2 = prop.bbox
            x1, y1 = max(x1 - self.pad_mod, 0), max(y1 - self.pad_mod, 0)
            x2, y2 = min(x2 + self.pad_mod, origin_width), min(
                y2 + self.pad_mod, origin_height
            )
            mask[y1:y2, x1:x2] = 255

        pad_image = pad_img_to_modulo(
            image,
            mod=self.pad_mod,
            square=self.pad_to_square,
            min_size=self.min_size,
        )
        pad_mask = pad_img_to_modulo(
            mask,
            mod=self.pad_mod,
            square=self.pad_to_square,
            min_size=self.min_size,
        )

        result = self.forward(pad_image, pad_mask)
        result = result[0:origin_height, 0:origin_width, :]

        original_pixel_indices = mask != 255
        result[original_pixel_indices] = image[:, :, ::-1][
            original_pixel_indices
        ]
        return result

    @torch.no_grad()
    def __call__(self, image, mask, resize_limit=512):
        """
        Args:
            image (np.ndarray): Image
            mask (np.ndarray): Mask
            resize_limit (int, optional): max size. Defaults to 512.

        Returns:
            np.ndarray: Inpainted Image
        """

        if resize_limit and max(image.shape) > resize_limit:
            origin_size = image.shape[:2]
            downsize_image = resize_max_size(image, size_limit=resize_limit)
            downsize_mask = resize_max_size(mask, size_limit=resize_limit)
            inpaint_result = self._pad_forward(downsize_image, downsize_mask)
            # only paste masked area result
            inpaint_result = cv2.resize(
                inpaint_result,
                (origin_size[1], origin_size[0]),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            inpaint_result = self._pad_forward(image, mask)

        original_pixel_indices = mask != 255
        inpaint_result[original_pixel_indices] = image[:, :, ::-1][
            original_pixel_indices
        ]

        return inpaint_result
