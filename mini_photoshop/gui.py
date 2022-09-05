import os
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit.web import cli as stcli
from streamlit_drawable_canvas import st_canvas

from mini_photoshop.model import InpaintModel, SalientModel
from mini_photoshop.utils import (
    get_size_limit,
    load_img,
    numpy_to_bytes,
    numpy_to_png_bytes,
)


inpaint_model = InpaintModel(device="cpu")
salient_model = SalientModel(model_name="u2net_lite", device="cpu")


def button_download(img, img_name, img_ext, transparent=False):
    if transparent:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        file_name = img_name + "_pts.png"
        data = numpy_to_png_bytes(img)
    else:
        img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB)
        file_name = img_name + "_pts" + img_ext
        data = numpy_to_bytes(img, img_ext[1:].upper())
    st.download_button(label="Download Image", data=data, file_name=file_name)


def salientcy():
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
    _, col1, col2, col3 = st.columns([0.1, 1, 1, 1])
    with col1:
        remove = st.button("Remove Background")
    with col2:
        blur = st.button("Blur Background")
    with col3:
        gray = st.button("Grayscale Background")

    if image_file is not None:
        img_name, img_ext = os.path.splitext(image_file.name)
        np_img, _ = load_img(image_file.read())

        mask = salient_model(np_img) * 255
        mask_inv = cv2.bitwise_not(mask)

        st_image = st.image(np_img, use_column_width=True)

        if remove:
            bg_removed = np_img.copy()
            bg_removed[mask == 0] = 255
            st_image.image(bg_removed)
            button_download(bg_removed, img_name, img_ext, transparent=True)

        if gray:

            def grayscale(x):
                return np.dot(x[:, :, :3], [0.299, 0.587, 0.114])

            bg_gray = grayscale(np_img)
            bg_gray = cv2.merge([bg_gray] * 3)

            fg = cv2.bitwise_and(np_img, np_img, mask=mask).astype(np.int32)
            bg = cv2.bitwise_and(bg_gray, bg_gray, mask=mask_inv).astype(
                np.int32
            )
            gray_img = cv2.add(fg, bg)
            st_image.image(gray_img)
            button_download(gray_img, img_name, img_ext)

        if blur:
            blur_bg = cv2.blur(np_img, (25, 25))

            fg = cv2.bitwise_and(np_img, np_img, mask=mask).astype(np.int32)
            bg = cv2.bitwise_and(blur_bg, blur_bg, mask=mask_inv).astype(
                np.int32
            )
            blur_img = cv2.add(fg, bg)
            st_image.image(blur_img)
            button_download(blur_img, img_name, img_ext)


def inpainting():
    is_draw = True
    canvas_width = 700
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

    _, col1, col2, _ = st.columns(4)
    with col1:
        auto = st.button("Auto Region")
    with col2:
        manual = st.button("Draw Region")

    if auto:
        is_draw = False
    if manual:
        is_draw = True

    if image_file is not None:
        img_name, img_ext = os.path.splitext(image_file.name)
        np_img, _ = load_img(image_file.read())
        resize_limit = get_size_limit(np_img.shape)

        if is_draw:
            h, w = np_img.shape[:2]
            ratio = canvas_width / w

            stroke_width = st.slider("Brush width: ", 10, 50, 30)

            canvas_result = st_canvas(
                stroke_width=stroke_width,
                background_image=Image.open(image_file),
                update_streamlit=False,
                width=np.floor(w * ratio),
                height=np.floor(h * ratio),
                drawing_mode="freedraw",
                key="canvas",
            )

            if canvas_result.image_data is not None:
                mask = 255 - canvas_result.image_data[:, :, 3]
                canvas_result.image_data = None
                mask[mask < 255] = 0
                if np.min(mask) < 255:
                    mask = 255 - mask

                    mask = cv2.resize(mask, dsize=(w, h))
                    np_img = np_img[:, :, ::-1]
                    inpainted = inpaint_model(np_img, mask, resize_limit)
                    st.image(inpainted, use_column_width=True)
                    button_download(inpainted, img_name, img_ext)

        else:
            mask = salient_model(np_img) * 255
            np_img = np_img[:, :, ::-1]
            inpainted = inpaint_model(np_img, mask, resize_limit)
            st.image(inpainted, use_column_width=True)
            button_download(inpainted, img_name, img_ext)


def main():
    st.set_page_config(
        page_title="Mini Photoshop Tool", page_icon=":film_frames:"
    )
    st.title("Mini Photoshop Tool")
    st.sidebar.subheader("Configuration")
    PAGES = {"Background Editor": salientcy, "Image Cleaner": inpainting}
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))

    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            """
            <h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png"
            alt="Streamlit logo" height="16">&nbsp by
            <a href="https://viblo.asia/u/tungbuitien">@tungbuitien</a></h6>
            """,
            unsafe_allow_html=True,
        )


def entry_point():
    if st._is_running_with_streamlit:
        main()
    else:
        abs_path = os.path.abspath(__file__)
        sys.argv = ["streamlit", "run", abs_path]
        sys.exit(stcli.main())


if __name__ == "__main__":
    entry_point()
