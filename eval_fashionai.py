# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:37:47 2018

@author: yfji
"""

import caffe
import json
import csv
import numpy as np
import matplotlib.pylab as plb
import os
import os.path as op
import cv2

caffe.set_mode_gpu()
net=caffe.Net('fashion_deploy.prototxt','/home/yfji/benchmark/Keypoint/fashionAI_key_points_train_20180227/train/train_fashion/models/fashion_iter_40000.caffemodel',caffe.TEST)
image_root=op.join(os.getcwd(), 'test')

input_size=net.blobs[net.inputs[0]].data.shape[2]
output_size=net.blobs[net.outputs[0]].data.shape[2]
stride=input_size/output_size

def find_peaks(fmap, thresh=0.4):
    map_left = np.zeros(fmap.shape)
    map_left[1:,:] = fmap[:-1,:]
    map_right = np.zeros(fmap.shape)
    map_right[:-1,:] = fmap[1:,:]
    map_up = np.zeros(fmap.shape)
    map_up[:,1:] = fmap[:,:-1]
    map_down = np.zeros(fmap.shape)
    map_down[:,:-1] = fmap[:,1:]
    
    peaks_binary = np.logical_and.reduce((fmap>=map_left, fmap>=map_right, fmap>=map_up, fmap>=map_down, fmap > thresh))
    peaks = np.hstack((np.nonzero(peaks_binary)[1].reshape(-1,1), np.nonzero(peaks_binary)[0].reshape(-1,1))) # note reverse
    peaks_with_score = [(x[0],x[1]) + (fmap[x[1],x[0]],) for x in peaks]
    print(peaks_with_score)
    return peaks_with_score


category_keypoints={
    'blouse':[0,1,2,3,4,5,6,9,10,11,12,13,14],
    'dress':[0,1,2,3,4,5,6,7,8,17,18],
    'skirt':[15,16,17,18],
    'outwear':[0,1,3,4,5,6,9,10,11,12,13,14],
    'trousers':[15,16,19,20,21,22,23]
}

def test_all():
    with open('test.csv', 'r') as f:
        reader=csv.reader(f)
    with open('pred.csv','w') as pred_f:
        pred_writer=csv.writer(pred_f)
    for ix,row in enumerate(reader):
        if ix==0:
            print(list(row))
            continue
        list_data=list(row)
        image_path=op.join(image_root,list_data[0])
        category=list_data[1]
        image=cv2.imread(image_path)
        scale=1.0*input_size/max(image.shape[0],image.shape[1])
        image=cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        pad_image=np.zeros((input_size,input_size,3),dtype=np.float32)
        pad_image+=128
        pad_image[:image.shape[0],:image.shape[1],:]=image
        pad_image=pad_image.transpose(2,0,1)[np.newaxis,:,:,:]
        pad_image=(pad_image-128)/0.5
        net.forward(**{net.inputs[0]:pad_image})
        
        output_blob=net.blobs[net.outputs[0]].data.squeeze()
        keypoints=category_keypoints[category]
        out_list=[]
        for i in range(24):
            if i in keypoints:
                heatmap=output_blob[i,:,:]
                peaks=find_peaks(heatmap)
                if len(peaks)==0:
                    out_list.append('-1_-1_-1')
                else:
                    peaks=sorted(peaks, key=lambda x:x[2])
                    peak=peaks[0]
                    raw_peak=[peak[0]*stride*1.0/scale,peak[1]*stride*1.0/scale]
                    out_list.append('%d_%d_1'%(int(raw_peak[0]),int(raw_peak[1])))
            else:
                out_list.append('-1_-1_-1')
        pred_str=','.join(out_list)
        print(image_path,pred_str)
        pred_row='%s,%s,%s'%(list_data[0],category,pred_str)
        pred_writer.writerow(pred_row)
    
    print('done')
    
def test_one():
    img_path='test/Images/trousers/0a179a6ba40d062c728eee5f48dd2665.jpg'
    image=cv2.imread(img_path)
     
    category='trousers'
    scale=1.0*input_size/max(image.shape[0],image.shape[1])
    raw_image=image.copy()
    image=cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    print(image.shape) 
    pad_image=np.zeros((input_size,input_size,3),dtype=np.float32)
    pad_image+=128
    pad_image[:image.shape[0],:image.shape[1],:]=image
    pad_image=pad_image.transpose(2,0,1)[np.newaxis,:,:,:]
    pad_image=(pad_image-128)/0.5
    net.forward(**{net.inputs[0]:pad_image})
    
    output_blob=net.blobs[net.outputs[0]].data.squeeze()
    keypoints=category_keypoints[category]
    out_list=[]
    for i in range(24):
        if i in keypoints:
            heatmap=output_blob[i,:,:]
            peaks=find_peaks(heatmap)
            print(peaks)
            if len(peaks)==0:
                out_list.append('-1_-1_-1')
            else:
                peaks=sorted(peaks, key=lambda x:x[2])
                peak=peaks[0]
                raw_peak=[peak[0]*stride*1.0/scale,peak[1]*stride*1.0/scale]
                out_list.append('%d_%d_1'%(int(raw_peak[0]),int(raw_peak[1])))
                cv2.circle(raw_image, (int(raw_peak[0]),int(raw_peak[1])), 6,(0,255,0),-1)
        else:
            out_list.append('-1_-1_-1')
    pred_str=','.join(out_list)
    print(img_path,pred_str)
    cv2.imshow('pred', raw_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__=='__main__':
    test_one()
    
        
        
    
    
    
    