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

use_bn=0
caffe.set_mode_gpu()
if use_bn:
    net=caffe.Net('fashion_deploy_bn.prototxt','/home/yfji/benchmark/Keypoint/fashionAI_key_points_train_20180227/train/train_fashion/models_bn/fashion_iter_100000.caffemodel',caffe.TEST)
else:
    net=caffe.Net('fashion_deploy.prototxt','/home/yfji/benchmark/Keypoint/fashionAI_key_points_train_20180227/train/train_fashion/models/fashion_iter_180000.caffemodel',caffe.TEST)
image_root=op.join(os.getcwd(), 'test')

input_size=net.blobs[net.inputs[0]].data.shape[2]
output_size=net.blobs[net.outputs[0]].data.shape[2]
stride=input_size/output_size

def find_peaks(fmap, thresh=0.1):
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
#    print(peaks_with_score)
    return peaks_with_score


category_keypoints={
    'blouse':[0,1,2,3,4,5,6,9,10,11,12,13,14],
    'dress':[0,1,2,3,4,5,6,7,8,17,18],
    'skirt':[15,16,17,18],
    'outwear':[0,1,3,4,5,6,9,10,11,12,13,14],
    'trousers':[15,16,19,20,21,22,23]
}

def test_all():
    with open('/home/yfji/benchmark/Keypoint/fashionAI_key_points_train_20180227/train/Annotations/train.csv','r') as tf:
        reader=csv.reader(tf)
        header=None
        for row in reader:
            header=list(row)
            break
    print(header)
    csv_name='pred_bn_iter_100000.csv' if use_bn else 'pred_iter_180000.csv'
    with open('test/test.csv', 'r') as f:
        reader=csv.reader(f)
        with open(csv_name,'w') as pred_f:
            pred_writer=csv.writer(pred_f, dialect='excel')
            for ix,row in enumerate(reader):
                if ix==0:
                    pred_writer.writerow(header)
                    continue
                list_data=list(row)
                image_path=op.join(image_root,list_data[0])
                category=list_data[1]
                image=cv2.imread(image_path)
                scale=1.0*input_size/max(image.shape[0],image.shape[1])
                image=cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).astype(np.float32)
#                image=cv2.resize(image, (input_size,input_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                pad_image=np.zeros((input_size,input_size,3),dtype=np.float32)
                pad_image+=128
                pad_image[:image.shape[0],:image.shape[1],:]=image
                pad_image=pad_image.transpose(2,0,1)[np.newaxis,:,:,:]
                pad_image=(pad_image-128)/256
                net.forward(**{net.inputs[0]:pad_image})
                
                output_blob=net.blobs[net.outputs[0]].data.squeeze().transpose(1,2,0)
                output_blob=cv2.resize(output_blob,(0,0),fx=stride,fy=stride, interpolation=cv2.INTER_CUBIC)
                keypoints=category_keypoints[category]
                out_list=[]
                for i in range(24):
                    if i in keypoints:
                        heatmap=output_blob[:,:,i]
                        heatmap=heatmap[:,:,np.newaxis]
                        peaks=find_peaks(heatmap)
                        if len(peaks)==0:
                            out_list.append('-1_-1_-1')
                        else:
                            peaks=sorted(peaks, key=lambda x:x[2], reverse=True)
                            peak=peaks[0]
                            raw_peak=[peak[0]*1.0/scale,peak[1]*1.0/scale]
                            out_list.append('%d_%d_1'%(int(raw_peak[0]),int(raw_peak[1])))
                    else:
                        out_list.append('-1_-1_-1')
                pred_str=','.join(out_list)
                print(image_path,pred_str)
                pred_row=[list_data[0],category]
                pred_row.extend(out_list)
                pred_writer.writerow(pred_row)
    
    print('done')
    
def test_one():
    category='trousers'
    img_path='test/Images/%s/0aea271ad540cb9293bc76487ab2cffe.jpg'%category
    image=cv2.imread(img_path)
    print('raw image shape',image.shape)
    scale=1.0*input_size/max(image.shape[0],image.shape[1])
    print('scale: %f'%scale)
    raw_image=image.copy()
    image=cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    print(image.shape) 
    pad_image=np.zeros((input_size,input_size,3),dtype=np.float32)
    pad_image+=128.0
    pad_image[:image.shape[0],:image.shape[1],:]=image
    pad_image=pad_image.transpose(2,0,1)[np.newaxis,:,:,:]
    pad_image=(pad_image-128)/256.0
    
    print(pad_image.shape)
    net.forward(**{net.inputs[0]:pad_image})
    
    output_blob=net.blobs[net.outputs[0]].data.squeeze().transpose(1,2,0)
    output_blob=cv2.resize(output_blob,(0,0),fx=stride,fy=stride, interpolation=cv2.INTER_CUBIC)
    print(output_blob.shape)
    keypoints=category_keypoints[category]
    out_list=[]
    for i in range(24):
        if i in keypoints:
            heatmap=output_blob[:,:,i]
            peaks=find_peaks(heatmap)
            print(peaks)
            if len(peaks)==0:
                out_list.append('-1_-1_-1')
            else:
                peaks=sorted(peaks, key=lambda x:x[2], reverse=True)
                peak=peaks[0]
                raw_peak=[peak[0]*1.0/scale,peak[1]*1.0/scale]
                out_list.append('%d_%d_1'%(int(raw_peak[0]),int(raw_peak[1])))
                cv2.circle(raw_image, (int(raw_peak[0]),int(raw_peak[1])), 6,(0,255,0),-1)
        else:
            out_list.append('-1_-1_-1')
    pred_str=','.join(out_list)
    print(img_path,pred_str)
    cv2.imshow('pred', raw_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def check_pred():
    csv_name='pred_iter_180000.csv' if use_bn else 'pred.csv'
    all_rows=[]
    with open('pred_bn.csv', 'r') as f:
        reader=csv.reader(f)
        for ix,row in enumerate(reader):
            if ix==0:
                print(list(row))
                continue
            all_rows.append(list(row))
    count=len(all_rows)
    random_order=np.random.permutation(count)
    for i in range(count):  
        list_data=all_rows[random_order[i]]
        image_path=op.join(image_root,list_data[0])
        category=list_data[1]
        keypoints=list_data[2:]            
        image=cv2.imread(image_path)
        for kpstr in keypoints:
            kp=kpstr.split('_')
            if kp[-1]==-1:
                continue
            cv2.circle(image, (int(kp[0]),int(kp[1])), 6,(0,255,0),-1)
        cv2.imshow('cloth', image)
        if cv2.waitKey()==27:
            break

if __name__=='__main__':
#    check_pred()
    test_all()
