import sys
sys.path.insert(0,'/data/xiaobing.wang/pingjun.li/yfji/Realtime_Multi-Person_Pose_Estimation-master/caffe_train-master/python')

import caffe
import csv
import numpy as np
import matplotlib.pylab as plb
import os
import os.path as op
import cv2

use_bn=0
caffe.set_mode_gpu()
caffe.set_device(1)
if use_bn:
    net=caffe.Net('fashion_deploy_bn.prototxt','/home/yfji/benchmark/Keypoint/fashionAI_key_points_train_20180227/train/train_fashion/models_bn/fashion_iter_100000.caffemodel',caffe.TEST)
else:
    net=caffe.Net('fashion_deploy.prototxt','../train/train_fashion/models/fashion_iter_90000.caffemodel',caffe.TEST)

image_root=op.join(os.getcwd(), './test')

input_size=net.blobs[net.inputs[0]].data.shape[2]
output_size=net.blobs[net.outputs[0]].data.shape[2]
stride=input_size/output_size

target_scale=0.8
scale_multipliers=[0.9,1.0,1.25,1.5]

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

def test_all():
    with open('../train/Annotations/train.csv','r') as tf:
        reader=csv.reader(tf)
        header=None
        for row in reader:
            header=list(row)
            break
    print(header)
    csv_name='pred_bn_iter_240000.csv' if use_bn else 'pred_iter_90000.csv'
    rows=[]
    with open('test/test.csv', 'r') as f:
        reader=csv.reader(f)
        for row in reader:
            rows.append(list(row))
    num_samples=len(rows)
    dummy_header=rows[0]
    print(dummy_header)
    rows=rows[1:]
    with open(csv_name,'w') as pred_f:
        pred_writer=csv.writer(pred_f, dialect='excel')
        pred_writer.writerow(header)
        for ix,row in enumerate(rows):
            image=cv2.imread(op.join(image_root,row[0]))
            category=row[1]
            keypoints_det=apply_multiscale(image, category,net)
            
            out_list=[]
            for k in range(24):
                out_list.append('%d_%d_1'%(keypoints_det[k,0],keypoints_det[k,1]))
            pred_row=[row[0],category]+out_list
            pred_writer.writerow(pred_row)
            print('%d/%d'%(ix+1,num_samples))
    print('done')

def test_one():
    category='trousers'
    img_path='test/Images/%s/0fe3b8126aed0858573ed0395adbca2b.jpg'%category
    raw_image=cv2.imread(img_path)
    keypoints=apply_multiscale(raw_image, category, net)
    for i in range(24):
        if i not in category_keypoints[category]:
            color=(255,0,0)
        else:
            color=(0,255,0)
        cv2.circle(raw_image, (int(keypoints[i,0]),int(keypoints[i,1])), 6, color,-1)
    print('done')
    cv2.imwrite('result_multiscale.jpg',raw_image)    
       
def apply_multiscale(image, category, net):
    final_keypoints=np.zeros((24,3),dtype=np.float32)
    cx,cy=input_size/2,input_size/2
    
    det_cnt=np.ones(24)
    for i,multiplier in enumerate(scale_multipliers):
        scale=target_scale*multiplier
        scale_image=cv2.resize(image, (0,0),fx=scale,fy=scale, interpolation=cv2.INTER_CUBIC)
        
        input_image=128*np.ones((input_size,input_size,3),dtype=np.float32)
        nw=min(input_image.shape[1],scale_image.shape[1])
        nh=min(input_image.shape[0],scale_image.shape[0])
        
        input_image[0:nh,0:nw,:]=image[0:nh,0:nw,:]
        
        keypoints=run_model_basic(input_image, category, net)
        det_index=keypoints[:,2]==1.0
        
        det_cnt[det_index]+=1

        keypoints[:,0]/=scale
        keypoints[:,1]/=scale
        final_keypoints[:,:2]+=keypoints[:,:2]
        final_keypoints[:,2]=np.maximum(final_keypoints[:,2],keypoints[:,2])
    final_keypoints[:,:2]/=det_cnt[:,np.newaxis]
    return final_keypoints
    
def run_model_basic(image, category, net):
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
    
if __name__=='__main__':
    test_one()