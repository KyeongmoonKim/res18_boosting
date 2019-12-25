import numpy as np
import configure as cf
import os
import cv2


def get_accuracy(outs, labels): #outs, labels are [batch_size, cls_um] np array
    cnt = 0
    for i in range(0, cf.BATCH_SIZE):
        x = np.argmax(outs[i])
        y = np.argmax(labels[i])
        if(x==y):
            cnt=cnt+1
    accuracy = cnt/cf.BATCH_SIZE
    return accuracy

def resize_image(img):
    (h, w) = img.shape[:2]
    max_len = max(h, w)
    if(max_len >= 224):
        if(h<w):
            ratio = h/w
            h_new = int(224 * ratio)
            w_new = 224
            img = cv2.resize(img, dsize=(h_new, w_new), interpolation=cv2.INTER_LINEAR)
        else:
            ratio = w/h
            h_new = 224
            w_new = int(224 * ratio)
            img = cv2.resize(img, dsize=(h_new, w_new), interpolation=cv2.INTER_LINEAR)
        #size rescaling
        #padding part
    (h, w) = img.shape[:2]
    horizon = 224-h
    vertical = 224-w
    #image is in center, can revise with random value on bottom
    bottom = int(horizon/2)
    top = horizon - bottom
    left = int(vertical/2)
    right = vertical-left
    ret = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return ret