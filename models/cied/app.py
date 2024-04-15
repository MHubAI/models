from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from io import BytesIO
import base64
import time
import logging

logging.basicConfig(filename="/logs/backend.log", level=logging.ERROR)


import numpy as np
from fastai.vision.all import *
import skimage
import PIL.Image
from PIL import ImageOps
from typing import Tuple, List, Dict
app = FastAPI()


crop_size = (256, 256)


def resize_img(img: Image.Image, small_ax: int = 1024) -> Image.Image:
    """Resize the input image.

    Args:
        img (Image): The input image to be resized.

    Returns:
        Image: The resized image.
    """
    scale_f = 1024 / min(img.size)
    return img.resize(
        (
            np.floor(img.size[0] * scale_f).astype(int),
            np.floor(img.size[1] * scale_f).astype(int),
        )
    )


def fix_bbox(
    bbox_orig: Tuple[int],
    img_shape: Tuple[int],
    minsize: int = 160,
    verbose: bool = False,
) -> Tuple[int]:
    """Standardize the bounding box for classification.

    Args:
        bbox_orig (Tuple[int]): Original bounding box coordinates.
        img_shape (Tuple[int]): Shape of the original image.
        minsize (int, optional): Minimum size for the bounding box. Defaults to 160.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        Tuple[int]: Standardized bounding box coordinates.
    """
    # Add margins to the detected object crop
    minr, minc, maxr, maxc = bbox_orig
    minr -= int(np.floor((maxr - minr) * 0.2))
    minc -= int(np.floor((maxc - minc) * 0.2))
    maxr += int(np.floor((maxr - minr) * 0.2))
    maxc += int(np.floor((maxc - minc) * 0.2))

    # Set the minimal size to the object crop
    dr = max(0, minsize - (maxr - minr))
    dc = max(0, minsize - (maxc - minc))
    minr -= dr // 2
    maxr += dr // 2
    minc -= dc // 2
    maxc += dc // 2

    # Make crop a square
    hr = maxr - minr
    hc = maxc - minc
    maxh = max(hr, hc)
    dr = maxh - hr
    dc = maxh - hc
    minr -= dr // 2
    maxr += dr // 2
    minc -= dc // 2
    maxc += dc // 2

    # Shift the expanded crop so it located within the image
    if verbose:
        print(img_shape)
        print(minr, maxr, minc, maxc)
    drmin = min(0, img_shape[0] - maxr)
    minr += drmin
    maxr = min(img_shape[0], maxr)

    drmax = min(0, minr)
    maxr -= drmax
    minr = max(0, minr)

    dcmin = min(0, img_shape[1] - maxc)
    minc += dcmin
    maxc = min(img_shape[1], maxc)

    dcmax = min(0, minc)
    maxc -= dcmax
    minc = max(0, minc)
    if verbose:
        print(minr, maxr, minc, maxc)

    return minr, minc, maxr, maxc


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the model on application startup."""

    global model_seg
    global model_clf_manuf
    global model_clf_model

    model_seg = load_learner("/models/segmentation.pkl")
    model_clf_manuf = load_learner("/models/classification_manuf.pkl")
    model_clf_model = load_learner("/models/classification_model.pkl")


def predict_class(img: Image.Image) -> Tuple[Image.Image, str]:
    """Predict the class of the input image.
    This function segments the input image to identify the device, defines a bounding box around it,
    and then classifies the device's manufacturer and model. 

    Args:
        img (Image): The input image for prediction.

    Returns:
        Tuple[Image, str]: A tuple containing the processed image and the prediction text.
    """
    try:
        img = ImageOps.exif_transpose(img)
        img_array = np.array(resize_img(img))

        # Inferring the segmentation mask
        res = model_seg.predict(img_array, with_input=True)

        mask = np.array(res[1])
        mask[mask != 1] = 0

        # Finding the largest connected component
        labeled_image, count = skimage.measure.label(mask, return_num=True)
        objects = skimage.measure.regionprops(labeled_image)
        if len(objects) == 0:
            return img, "No device has been detected"
        object_areas = [obj["area"] for obj in objects]
        max_idx = np.argmax(object_areas)
        obj = objects[max_idx]

        # Defining the object crop
        minr, minc, maxr, maxc = fix_bbox(obj["bbox"], img_array.shape)
        crop = img_array[minr:maxr, minc:maxc]
        crop = PIL.Image.fromarray(crop).resize(crop_size)

        # Cropping, resizing and generating output for the frontend
        res_manuf = model_clf_manuf.predict(crop)
        pred_manuf = res_manuf[0]
        conf_manuf = np.max(np.array(res_manuf[2]))

        res_model = model_clf_model.predict(crop)
        pred_model = res_model[0]
        conf_model = np.max(np.array(res_model[2]))

        prediction = f"Manufacturer: {pred_manuf} (Confidence: {conf_manuf:.2}). Model: {pred_model} (Confidence: {conf_model:.2})."
    except Exception as e:
        logging.error(f"{e}")
        crop = img.convert("L")
        prediction = "Unhandled error has occured, refer to the logs for details."
    return crop, prediction


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    This function is triggered upon receiving a request from the frontend. It decodes input, calls the prediction model and sends back the response.
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        processed_image, text = predict_class(image)
        buffered = BytesIO()
        processed_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return {"image": img_str, "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
