import configure as cf
import cv2
import copy
import os
import random
import numpy as np
import csv

#vgg_mean = [103.939, 116.779, 123.68]
#data read and rgb -> bgr
#all data 
class Data:
    def __init__(self, img_dir, cls_dir, oversampling=False): #img_dir : jpeg images, cls_dir : xml(voc format)
        self.img_dir = img_dir
        self.cls_dir = cls_dir
        self.cnt = 0
        self.name_list = []
        self.label_list = []
        self.num = 0
        f = open(cls_dir, 'r')
        rd = csv.reader(f)
        cnt_list = [0]*cf.CLS_NUM
        print(cnt_list)
        for line in rd:
            if(self.num%100==0):
                print(str(self.num)+' images loaded')
            self.name_list.append(line[0])
            self.label_list.append(int(line[1]))
            self.num = self.num+1 
            cnt_list[int(line[1])] = cnt_list[int(line[1])] + 1
       #It will be used to gurantee all sample to be used
        print("There are " +str(self.num)+" images")
        if(oversampling): #oversampliing the samples with the number of cf.CLS_NUM * maximum 
            maximum = max(cnt_list)
            print("start oversampling..")
            temp_name_list = []
            temp_label_list = []
            temp_num = 0
            for i in range(0, self.num):
                if(i%1000==0):
                    print("(oversampling) "+str(i)+ " images loaded")
                curr_label = int(self.label_list[i])
                time = max(int((maximum * cf.OVER_RATIO)/cnt_list[curr_label]), 1) #ith item will be inserted with counts of the 'time'
                for j in range(0, time):
                    temp_name_list.append(self.name_list[i])
                    temp_label_list.append(self.label_list[i])
                    temp_num = temp_num+1
            #update
            self.name_list = temp_name_list
            self.label_list = temp_label_list
            self.num = temp_num
            print(len(self.name_list))
            print(len(self.label_list))
            print(self.num)
            cnt_list = [0] * cf.CLS_NUM
            for i in range(0, self.num):
                cnt_list[int(self.label_list[i])] = cnt_list[int(self.label_list[i])] + 1
            print(cnt_list)
            print(sum(cnt_list))
        self.rand_idx = random.sample(range(self.num), self.num) #non duplicate idx list
        print(len(self.rand_idx))
        f.close()
        
    def get_batch(self, gurantee=True, resizing=False): #return batch and cls
        #print(self.cnt)
        if(gurantee==True):
            if(self.cnt+cf.BATCH_SIZE-1 >= self.num): #used all sample one times
                self.rand_idx = random.sample(range(self.num), self.num)
                self.cnt = 0
            batch_idx_list = self.rand_idx[self.cnt:self.cnt+cf.BATCH_SIZE]
            self.cnt = self.cnt+cf.BATCH_SIZE
        else:
            batch_idx_list = random.sample(range(self.num), cf.BATCH_SIZE)
        #print(self.cnt)
        batch_img_list = []
        batch_lab_list = []
        for i in range(0, cf.BATCH_SIZE):
            try:
                img = cv2.imread(self.img_dir+self.name_list[batch_idx_list[i]]+'.jpg')
            except:
                print(i)
                print(batch_idx_list[i])
            if(resizing):
                try:
                    img = self.resize_image(img)
                except:
                    print(self.img_dir+self.name_list[batch_idx_list[i]]+'.jpg')
                    assert False
            try:
              (h, w) = img.shape[:2]
            except:
              print("here")
              print(i)
              print(batch_idx_list[i])
              print(self.name_list[batch_idx_list[i]])
              assert False
            test = (h==224)and(w==224)
            if not(test):
              print("name : "+self.name_list[batch_idx_list[i]])
              print("h, w : "+str(h) +" "+str(w))
              assert test 
            img = img.reshape(1, 224, 224, 3)
            label = self.one_hot_encoding(self.label_list[batch_idx_list[i]])
            batch_img_list.append(img)
            batch_lab_list.append(label)
        batch_img_list = np.concatenate(batch_img_list)
        batch_img_list = batch_img_list.astype('float32')
        batch_lab_list = np.concatenate(batch_lab_list)
        batch_name_list = []
        for i in range(0, cf.BATCH_SIZE):
            #print(str(i) + " "+ self.name_list[batch_idx_list[i]])
            batch_name_list.append(self.name_list[batch_idx_list[i]])
        #print(batch_img_list)
        #print(batch_lab_list)
        return batch_img_list, batch_lab_list, batch_name_list
        
    def resize_image(self, img):
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
    def one_hot_encoding(self, idx):
        ret = [0.0] * cf.CLS_NUM
        ret[int(idx)] = 1.0
        ret = np.array(ret)
        ret = ret.reshape(1, cf.CLS_NUM)
        #print(ret.shape)
        return ret
    
    