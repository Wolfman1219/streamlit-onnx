from RT.models.utils import blob
from RT.models.torch_utils import det_postprocess
from RT.models.cudart_api import TRTEngine
from RT.models import EngineBuilder
import torch
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

# except:
# st.write("ooops! packages did not loaded")

def ImageBox(image, new_shape=(640, 640), color=(0, 0, 0)):

    width, height, channel = image.shape

    ratio = min(new_shape[0] / width, new_shape[1] / height)

    new_unpad = int(round(height * ratio)), int(round(width * ratio))

    dw, dh = (new_shape[0] - new_unpad[0])/2, (new_shape[1] - new_unpad[1])/2

    if (height, width) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, ratio, (dw, dh)


def run_tensorrt(enggine_path, image):
    enggine = TRTEngine(enggine_path)
    
    height, width, channels = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cache_image, ratio, dwdh = ImageBox(image)

    tensor = blob(cache_image, return_seg=False)
    tensor = torch.asarray(tensor)

    results = enggine(tensor)
    dwdh = np.array(dwdh * 2, dtype=np.float32)

    bboxes, scores, labels = det_postprocess(results)
    bboxes = (bboxes-dwdh)/ratio

    for (bbox, score, label) in zip(bboxes, scores, labels):
        bbox = bbox.round().astype(np.int32).tolist()
        cv2.rectangle(image, (bbox[0],bbox[1]) , (bbox[2],bbox[3]) , (255,255,255), 1)
    return image