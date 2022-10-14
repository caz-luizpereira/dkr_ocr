

# import streamlit as st
# dependencies
from IPython.display import Image
from matplotlib import pyplot as plt

import cv2
import argparse
import sys
import numpy as np
import os
import os.path
from random import randint
from os import rename
from PIL import Image
import pandas as pd
import time

confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image


#=====================  SPLIT CLASSES  ============================
def split_classes(classesFile):
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

#===================================================================

# ================= Get the names of the output layers =============
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#=======================================================================

#=====================  DEFINING NETWORK  ============================
def network (modelConfiguration, modelWeights):
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

#======================================================================

def drawPred(classId, conf, left, top, right, bottom, frame,classes):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, ), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    top = max(top, labelSize[1])
    # cv2.rectangle(frame, (left, top - round(2*labelSize[1])), (left + round(2*labelSize[0]), top + baseLine), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine-10),    (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

#====================================================================

def postprocess(frame, outs,classes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    labeled = []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    print(indices)
    if(len (indices)<=0):
        cropped = np.zeros(3)
        left=0
        top=0 
        right=0
        bottom=0
        labeled="Nothing"


    else:
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            label = str.upper((classes[classIds[i]]))
            labeled.append( classes[classIds[i]])
            # calculate bottom and right
            bottom = top + height
            right = left + width
            print("top:",top)
            print("b:",bottom)
            print("l:",left)
            print("r:",right)
            if top<0:
                top=0
            if left<0:
                left=0
            if right>frameWidth:
                right=frameWidth
            if bottom>frameHeight:
                bottom=frameHeight
            cropped = frame[top:bottom, left:right].copy()
            print("cr:",cropped)
            # drawPred
            drawPred(classIds[i], confidences[i], left, top, right, bottom, frame,classes)

    return cropped,left, top, right, bottom,labeled


def load_net():
    modelConfiguration_3 = "Weights/char/yolov4_custom.cfg"
    modelWeights_3 = "Weights/char/yolov4_custom_last.weights"
    classesFile_3 = "Weights/char/obj.names"
    

    print ("------------- DEFINING YOLO PATHS -----------------")

    #======================  split classes  ===========================

    classes_3 = split_classes(classesFile_3)

    print ("------------- SPLIT ALL CLASSES  -----------------")

    #=====================  LOADING YOLO ===========================

    net_3 = network(modelConfiguration_3, modelWeights_3)

    print ("------------- LOADING ALL YOLO's -----------------")
    return classes_3,net_3


def yolo3(input_image):
        
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net_3.setInput(blob)

        # Runs the forward pass to get output of the output layers
        t5 = time.time()
        outs_3 = net_3.forward(getOutputsNames(net_3))
        t6 = time.time()
        tx3=t6-t5
        print("time 3: ",tx3)
        boxes, confidences, class_IDs = [], [], []
        H, W = input_image.shape[:2]
        for output in outs_3:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")
                    x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_IDs.append(classID)


        # Apply non-max suppression to identify best bounding box
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        xmin, xmax, ymin, ymax, labels = [], [], [], [], []
        xmi,xma,ymi,yma,lc=[],[],[],[],[]
        xc,xcm,yc,ycm,ln=[],[],[],[],[]
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                xmin.append(x)
                ymin.append(y)
                xmax.append(x+w)
                ymax.append(y+h)
                label = str.upper((classes_3[class_IDs[i]]))
                print("************")

                print('label ',label)
                if (label == str('A') or label == str('B')or label == str('C')
                or label == str('D')or label == str('E')or label == str('F')
                or label == str('G')or label == str('H')or label == str('I')
                or label == str('J')or label == str('K')or label == str('L')
                or label == str('M')or label == str('N')or label == str('O')
                or label == str('P')or label == str('Q')or label == str('R')
                or label == str('S')or label == str('T')or label == str('U')or label == str('V')
                or label == str('W')or label == str('X')or label == str('Y')or label == str('Z')):
                    xmi.append(x)
                    ymi.append(y)
                    xma.append(x+w)
                    yma.append(y+h)
                    lc.append(label)
                    print("xmi ",xmi,ymi,xma,yma)

                if (label == str('0') or label == str('1')or label == str('2')or label == str('3')
                or label == str('4')or label == str('5')or label == str('6')or label == str('7')
                or label == str('8')or label == str('9')):
                    xc.append(x)
                    yc.append(y)
                    xcm.append(x+w)
                    ycm.append(y+h)
                    ln.append(label)
                    print("xc ",xc,yc,xcm,ycm)

                print("************")
        boxes = pd.DataFrame({"xmin":   xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
        char = pd.DataFrame({"xmi": xmi, "ymi": ymi, "xma": xma, "yma": yma, "Label": lc})
        num = pd.DataFrame({"xmi": xc, "yc": yc, "xca": xcm, "yca": ycm, "Label": ln})
        
        final = pd.concat([char, num])
        final.sort_values(by=['xmi'], inplace=True)
        res=final[final.columns[4]]

        # Add a layer on top on a detected object 
        LABEL_COLORS = [0, 255, 0]
        image_with_boxes = input_image.astype(np.float64)
        for _, (xmin, ymin, xmax, ymax) in boxes.iterrows():
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += (np.random.randint(0, 255, size=(3),dtype="uint8"))
            cv2.rectangle(image_with_boxes,(xmin,ymin),(xmax,ymax),(0,0,0),1)
            # cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine-10),    (255, 255, 255), cv2.FILLED)
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2


        tx=tx3

        print("OVERALL TIME: ",tx)
        characters = image_with_boxes.astype(np.uint8)

        d1 = pd.DataFrame({"label":res})
       
        d2 = d1.transpose()
        d3 = d2.values.tolist()
        flatten_mat=[]
        for sublist in d3:
            for val in sublist:
                print(val)
                flatten_mat.append(val)


        lp_num = ''.join(map(str,flatten_mat))
        print("LP NUM: ",lp_num)
        cv2.imwrite('sample3.jpg', image_with_boxes.astype(np.uint8))
        

        return characters,lp_num
        #break

classes_3,net_3 = load_net()