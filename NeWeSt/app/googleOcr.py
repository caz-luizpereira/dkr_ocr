import os
import io
import json
from google.cloud import vision
from google.cloud.vision_v1 import types
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import time
from base64 import b64encode
from IPython.display import Image
from PIL import Image as im
#import functions as func
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="symmetric-ion-354715-4f4728096372.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="omega-iterator-355710-02b3d32ee01b.json"


def requestOCR(url, chave, imgpath):
  imgdata = makeImageData(imgpath)
  response = requests.post(url, 
                           data = imgdata, 
                           params = {'key': chave}, 
                           headers = {'Content-Type': 'application/json'})
  return response

def makeImageData(imgpath):
    img_req = None
    #with open(imgpath, 'rb') as f:
    ctxt = b64encode(imgpath).decode() #b64encode(f.read()).decode()
    img_req = {
        'image': {
            'content': ctxt
        },
        'features': [{
            'type': 'DOCUMENT_TEXT_DETECTION',
            'maxResults': 1
        }]
    }
    return json.dumps({"requests": img_req}).encode()

def gen_cord(result):
  cord_df = pd.DataFrame(result['boundingPoly']['vertices'])
  x_min, y_min = np.min(cord_df["x"]), np.min(cord_df["y"])
  x_max, y_max = np.max(cord_df["x"]), np.max(cord_df["y"])
  return result["description"], x_max, x_min, y_max, y_min



def google_Master_this_shit(image):
    image = cv2.resize( image, (300,120), interpolation = cv2.INTER_CUBIC)
    image = cv2.medianBlur(image, 5)

    ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
    chave = "" #chave google
    #cv2.imshow("img",image)
    #cv2.waitKey(0)
    pil_im = im.fromarray(image)
    b = io.BytesIO()
    pil_im.save(b, 'png')
    im_bytes = b.getvalue()
    result = requestOCR(ENDPOINT_URL, chave, im_bytes)
    if result.status_code != 200 or result.json().get('error'):
        print ("Error")
    else:
        result = result.json()['responses'][0]['textAnnotations']


    #for index in range(len(result)):
    text, x_max, x_min, y_max, y_min = gen_cord(result[0])
    return text
