import requests
import base64
import io
import cv2
from PIL import Image
import numpy as np


def predict_image(image, api_key, url, idx):
    retval, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer)
    img_str = img_str.decode("ascii")

    # Construct the URL
    upload_url = "".join([
        url,
        "?api_key=",
        api_key,
        "&name=",
        str(idx),
        ".jpg"
    ])

    # POST to the API
    r = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })

    json = r.json()

    predictions = json["predictions"]
    formatted_predictions = []
    classes = []

    for pred in predictions:
        formatted_pred = [pred["x"], pred["y"], pred["width"], pred["height"], pred["confidence"]]
        formatted_predictions.append(formatted_pred)
        classes.append(pred["class"])

    #print(formatted_predictions)

    return formatted_predictions, classes
