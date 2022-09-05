from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def ceil_modulo(x, mod):
    """Get the smallest integer divisible by modulo,
    greater than or equal to x
    """
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def numpy_to_png_bytes(image_numpy):
    """Convert numpy array to byte (png)

    Args:
        image_numpy (np.ndarray): image

    Returns:
        byte: image buffer (png)
    """
    im = Image.fromarray(image_numpy, mode="RGBA")
    datas = im.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    im.putdata(newData)

    buffered = BytesIO()
    im.save(buffered, format="PNG")
    img_data = buffered.getvalue()
    return img_data


def numpy_to_bytes(image_numpy: np.ndarray, ext: str):
    """Convert numpy array to bytes

    Args:
        image_numpy (np.ndarray): image
        ext (str): extention of byte data

    Returns:
        bytes: image buffer
    """
    data = cv2.imencode(
        f".{ext}",
        image_numpy,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            100,
            int(cv2.IMWRITE_PNG_COMPRESSION),
            0,
        ],
    )[1]
    image_bytes = data.tobytes()
    return image_bytes


def load_img(img_bytes, gray: bool = False):
    """Load np image from image buffer

    Args:
        img_bytes (bytes): Image buffer
        gray (bool, optional): convert to grayscale. Defaults to False.

    Returns:
        np.ndarray: Image
    """
    alpha_channel = None
    nparr = np.frombuffer(img_bytes, np.uint8)
    if gray:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    else:
        np_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if len(np_img.shape) == 3 and np_img.shape[2] == 4:
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGRA2RGB)
        else:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    return np_img, alpha_channel


def norm_img(np_img):
    """Normalize image

    Args:
        np_img (np.ndarray): input image

    Returns:
        np.ndarray: normalized image
    """
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img


def resize_max_size(np_img, size_limit):
    """Resize image

    Args:
        np_img (np.ndarray): input image
        size_limit (int): max size

    Returns:
        np.ndarray: _description_
    """
    # Resize image's longer size to size_limit
    # if longer size larger than size_limit
    h, w = np_img.shape[:2]
    if max(h, w) > size_limit:
        ratio = size_limit / max(h, w)
        new_w = int(w * ratio + 0.5)
        new_h = int(h * ratio + 0.5)
        return cv2.resize(
            np_img, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC
        )
    else:
        return np_img


def pad_img_to_modulo(
    img: np.ndarray,
    mod: int,
    square: bool = False,
    min_size: Optional[int] = None,
):
    """Padding image

    Args:
        img (np.ndarray): Image
        mod (int): modulo value
        square (bool, optional): padding image to square. Defaults to False.
        min_size (Optional[int], optional): min size. Defaults to None.

    Returns:
        np.ndarray: Image after padding
    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


def get_size_limit(img_shape):
    """Define max size for image base on shape

    Args:
        img_shape (tupple): image shape

    Returns:
        int: max size
    """
    size = max(img_shape)
    if size > 1024:
        size_limit = 1024
    elif size > 512:
        size_limit = 512
    elif size > 256:
        size_limit = 256
    else:
        size_limit = None
    return size_limit
