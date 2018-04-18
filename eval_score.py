# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:31:19 2018

@author: yfji
"""

import caffe
import csv
import numpy as np
import matplotlib.pylab as plb
import os
import os.path as op
import cv2

use_bn=0
caffe.set_mode_gpu()
caffe.set_device(0)

if use_bn:
    net=caffe.Net('fashion_deploy_bn.prototxt','/home/yfji/benchmark/Keypoint/fashionAI_key_points_train_20180227/train/train_fashion/models_bn/fashion_iter_280000.caffemodel',caffe.TEST)
else:
    net=caffe.Net('fashion_deploy.prototxt','/home/yfji/benchmark/Keypoint/fashionAI_key_points_train_20180227/train/train_fashion/models/fashion_iter_90000.caffemodel',caffe.TEST)
image_root=op.join(os.getcwd(), 'test')

input_size=net.blobs[net.inputs[0]].data.shape[2]
output_size=net.blobs[net.outputs[0]].data.shape[2]
stride=input_size/output_size

print(input_size,output_size)

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
    return peaks_with_score
    
category_keypoints={
    'blouse':[0,1,2,3,4,5,6,9,10,11,12,13,14],
    'dress':[0,1,2,3,4,5,6,7,8,17,18],
    'skirt':[15,16,17,18],
    'outwear':[0,1,3,4,5,6,9,10,11,12,13,14],
    'trousers':[15,16,19,20,21,22,23]
}

def euclideanDistance(pt1, pt2):
    dist_vec=np.subtract(pt1,pt2)
    return np.sqrt(np.sum(dist_vec**2))

def run_model(row, net):
    image_root='/home/yfji/benchmark/Keypoint/fashionAI_warm_up_train_20180222/train/'
    image_path=op.join(image_root,row[0])
    category=row[1]
    image=cv2.imread(image_path)
    scale=1.0*input_size/max(image.shape[0],image.shape[1])
    image=cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    pad_image=128*np.ones((input_size,input_size,3),dtype=np.float32)
    pad_image[:image.shape[0],:image.shape[1],:]=image
    pad_image=pad_image.transpose(2,0,1)[np.newaxis,:,:,:]
    pad_image=(pad_image-128)/256.0
    net.forward(**{net.inputs[0]:pad_image})
    
    output_blob=net.blobs[net.outputs[0]].data.squeeze().transpose(1,2,0)
    output_blob=cv2.resize(output_blob,(0,0),fx=stride,fy=stride, interpolation=cv2.INTER_CUBIC)
    keypoints=category_keypoints[category]
    
    keypoints_det=-1*np.ones((24,3),dtype=np.float32)
    
    ltx=1e4;lty=1e4;rbx=0;rby=0
    for i in range(24):
        heatmap=output_blob[:,:,i]
        heatmap=heatmap[:,:,np.newaxis]
        peaks=find_peaks(heatmap)
        if len(peaks)>0:
            peaks=sorted(peaks, key=lambda x:x[2], reverse=True)
            peak=peaks[0]
            raw_peak=[peak[0]*1.0/scale,peak[1]*1.0/scale,1.0]
            keypoints_det[i]=np.asarray(raw_peak)
            if i in keypoints:
                ltx=min(ltx,raw_peak[0])
                lty=min(lty,raw_peak[1])
                rbx=max(rbx,raw_peak[0])
                rby=max(rby,raw_peak[1])
    center=[0.5*(ltx+rbx),0.5*(lty+rby)]
    arr=np.zeros(24)
    arr[keypoints]=1
    keypoints_det[arr==0,0]=center[0]
    keypoints_det[arr==0,1]=center[1]
    keypoints_det[arr==0,2]=0
    return keypoints_det
    
def criterion(category, keypoints_gt, keypoints_det):
    norm_dist=0
    if category in ['blouse','outwear','dress']:
        pt1=keypoints_gt[5,:2]
        pt2=keypoints_gt[6,:2]
        norm_dist=euclideanDistance(pt1,pt2)
    elif category in ['trousers','skirt']:
        pt1=keypoints_gt[15,:2]
        pt2=keypoints_gt[16,:2]
        norm_dist=euclideanDistance(pt1,pt2)
    else:
        raise Exception('Unknown type')
    scores=[]
    if norm_dist==0:
        return []
    for k in range(keypoints_gt.shape[0]):
        if keypoints_gt[k,-1]==1:
            dist=euclideanDistance(keypoints_gt[k],keypoints_det[k])
            scores.append(1.0*dist/norm_dist)
    return scores
    
def main():   
    rows=[]
    csv_name='/home/yfji/benchmark/Keypoint/fashionAI_warm_up_train_20180222/train/Annotations/annotations.csv'
    with open(csv_name,'r') as f:
        reader=csv.reader(f)
        for row in reader:
            rows.append(list(row))
    header=rows[0]
    rows=rows[1:]
    num_samples=len(rows)
    print(header, num_samples)
    
    score_sum=0.0
    cnt=0
    random_order=np.random.permutation(np.arange(num_samples))
    for ix in range(num_samples):
        if ix==5000:
            break
        row=rows[random_order[ix]]
        keypoints_det=run_model(row, net)
        keypoints_gt=-1*np.ones((24,3),dtype=np.float32)
        category=row[1]
        for k, kpstr in enumerate(row[2:]):
            kps=kpstr.split('_')
            keypoints_gt[k]=np.asarray(list(map(float,kps)))
        scores=criterion(category, keypoints_gt, keypoints_det)
        score_avg=0
        if len(scores)>0:
            score_avg=1.0*sum(scores)/len(scores)
            cnt+=1
        print('[%d/%d]:%f'%(ix,num_samples,score_avg))
        score_sum+=score_avg
    print('Total score: %f'%(score_sum/cnt))
    print('done')
        

if __name__=='__main__':
    main()
    
    
