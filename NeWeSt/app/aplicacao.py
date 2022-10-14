import os
import string
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
from matplotlib import pyplot as plt
# import pytesseract
# %matplotlib inline
#import functions as func
import googleOcr as google
from PIL import Image
import base64
import io
import numpy as np
from ocrNew import yolo3

def load_models():
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file('pipeline.config')
    detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    # ckpt.restore('Tensorflow/workspace/models/matriculas/export/checkpoint/checkpoint')
    ckpt.restore('ckpt-12').expect_partial()

    #labels = [{'name':'descartavel', 'id':1},{'name':'util', 'id':2},{'name':'parafusos', 'id':3}]
    labels = [{'name': 'matricula', 'id': 1}]

    with open('label_map.pbtxt', 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')
    ##########
    category_index = label_map_util.create_category_index_from_labelmap(
        'label_map.pbtxt')

    return category_index, detection_model


def detect_fn1(image_desc, detection_model_desc):
    return (detection_model_desc.postprocess(detection_model_desc.predict(detection_model_desc.preprocess(image_desc)[0], detection_model_desc.preprocess(image_desc)[1]), detection_model_desc.preprocess(image_desc)[1]))


def load_models_descartaveis():

    # Load pipeline config and build a detection model

    configs = config_util.get_configs_from_pipeline_file(
        'detetor_descartaveis/pipeline.config')

    detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)

    # Restore checkpoint

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)

    # ckpt.restore('Tensorflow/workspace/models/matriculas/export/checkpoint/checkpoint')

    ckpt.restore('detetor_descartaveis/ckpt-11').expect_partial()

    labels = [{'name': 'descartavel', 'id': 1}, {
        'name': 'util', 'id': 2}, {'name': 'parafusos', 'id': 3}]

    #labels = [{'name':'matricula','id':1}]

    with open('detetor_descartaveis/label_map.pbtxt', 'w') as f:

        for label in labels:

            f.write('item { \n')

            f.write('\tname:\'{}\'\n'.format(label['name']))

            f.write('\tid:{}\n'.format(label['id']))

            f.write('}\n')

    ##########

    category_index = label_map_util.create_category_index_from_labelmap(
        'detetor_descartaveis/label_map.pbtxt')

    return category_index, detection_model


def load_modelsTun():
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(
        'Tunisia/pipeline.config')
    detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)

    ckpt.restore('Tunisia/ckpt-21').expect_partial()

    labels = [{'name': 'matricula', 'id': 1}]

    with open('Tunisia/label_map.pbtxt', 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')
    ##########
    category_index = label_map_util.create_category_index_from_labelmap(
        'Tunisia/label_map.pbtxt')

    return category_index, detection_model


def mainTun(img, detections):  # (img,category_index,detection_model):

    image_64_decode = base64.b64decode(img)

    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    ######
    import numpy as np

    image = Image.open(io.BytesIO(image_64_decode))

    image_np = np.array(image)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    detection_threshold = 0.2
    image = image_np_with_detections
    scores = list(filter(lambda x: x > detection_threshold,
                  detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    width = image.shape[1]
    height = image.shape[0]
    # Apply ROI filtering and OCR
    matriculas_detected = []

    for idx, box in enumerate(boxes):

        roi = box*[height, width, height, width]

        region = image[int(roi[0])-3:int(roi[2])+3,
                       int(roi[1])-3:int(roi[3])+3]
        matriculas_detected.append(region)

        final = google.google_Master_this_shit(region)

    return final


# (img,category_index,detection_model):
def mainGoogle(img, detections, category_index_desc, detection_model_desc):
    todas_Matriculas = []
    image_64_decode = base64.b64decode(img)

    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    ######
    import numpy as np

    image = Image.open(io.BytesIO(image_64_decode))

    image_np = np.array(image)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(
        np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    detection_threshold = 0.2
    image = image_np_with_detections
    scores = list(filter(lambda x: x > detection_threshold,
                  detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    width = image.shape[1]
    height = image.shape[0]
    # Apply ROI filtering and OCR
    matriculas_detected = []

    for idx, box in enumerate(boxes):

        roi = box*[height, width, height, width]

        region = image[int(roi[0])-3:int(roi[2])+3,
                       int(roi[1])-3:int(roi[3])+3]
        matriculas_detected.append(region)

        image_np_desc = np.array(region)

        input_tensor_desc = tf.convert_to_tensor(
            np.expand_dims(image_np_desc, 0), dtype=tf.float32)

        detections_desc = detect_fn1(input_tensor_desc, detection_model_desc)

        num_detections_desc = int(detections_desc.pop('num_detections'))

        detections_desc = {key: value[0, :num_detections_desc].numpy()

                           for key, value in detections_desc.items()}

        detections_desc['num_detections'] = num_detections_desc

        # detection_classes should be ints.

        detections_desc['detection_classes'] = detections_desc['detection_classes'].astype(
            np.int64)

        label_id_offset_desc = 1

        image_np_with_detections_desc = image_np_desc.copy()

        detection_threshold_desc = 0.2

        image_desc = image_np_with_detections_desc

        scores_desc = list(filter(
            lambda x: x > detection_threshold_desc, detections_desc['detection_scores']))

        boxes_desc = detections_desc['detection_boxes'][:len(scores_desc)]

        classes_desc = detections_desc['detection_classes'][:len(scores_desc)]

        width_desc = image_np_with_detections_desc.shape[1]

        height_desc = image_np_with_detections_desc.shape[0]

        for i in range(len(classes_desc)):

            # if classes_desc[i]==0:

            #    y,x,h,w = boxes_desc[i]

            #    image_desc = cv2.rectangle(image_desc, (int(x*width_desc), int(y*height_desc)), (int(w*width_desc), int(h*height_desc)), (255, 255, 255), -1)

            if classes_desc[i] == 1:

                box_desc = boxes_desc[i]

                roi_desc = box_desc*[height_desc,
                                     width_desc, height_desc, width_desc]

                region_desc = image_desc[int(
                    roi_desc[0])-3:int(roi_desc[2])+3, int(roi_desc[1]):int(roi_desc[3])]

                pass

        final = google.google_Master_this_shit(region_desc)        
        todas_Matriculas.append(final)

    return todas_Matriculas


def mainGeral(img,detections):#(img,category_index,detection_model):

    image_64_decode = base64.b64decode(img) 

    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    

    
    ######
    import numpy as np

    image = Image.open(io.BytesIO(image_64_decode))

    image_np = np.array(image)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections


    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    detection_threshold = 0.2
    image = image_np_with_detections
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]


    width = image.shape[1]
    height = image.shape[0]
    # Apply ROI filtering and OCR
    matriculas_detected=[]

    for idx, box in enumerate(boxes):

        roi = box*[height, width, height, width]

        region = image[int(roi[0])-3:int(roi[2])+3,int(roi[1])-3:int(roi[3])+3]
        matriculas_detected.append(region)
        cropped,ocrfinal=yolo3(region)
        #final = google.google_doesnt_Master_this_shit(region)

    return ocrfinal
