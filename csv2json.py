# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:11:42 2018

@author: JYF
"""

import os
import json
import csv
import cv2
import PIL.Image as PI
import os.path as op
import numpy as np

images_root=op.join(os.getcwd(),'../')
annot_dir=os.getcwd()
net_input_side=368

def parse_json():
    with open(op.join(annot_dir,'train.csv'),'r') as f:
        reader=csv.reader(f)
        for ix,row in enumerate(reader):
            if ix==0:
                print(list(row))
                continue
            list_data=list(row)
            image_path=op.join(images_root,list_data[0])
            category=list_data[1]
            keypoints=list_data[2:]
            image=cv2.imread(image_path)
            for kp_str in keypoints:
                keypoint=kp_str.split('_')
                x=int(keypoint[0])
                y=int(keypoint[1])
                visible=int(keypoint[2])
                if visible==-1:
                    continue
                color=(0,255,0)
                if visible==0:
                    color=(0,0,255)
                cv2.circle(image, (x,y), 3, (0,0,255), -1)
    #            cv2.imshow('clothes',image)
    #            cv2.waitKey()
            cv2.imshow('clothes',image)
            if cv2.waitKey()==27:
                break
        cv2.destroyAllWindows()

def csv2json():
    with open(op.join(annot_dir,'train.csv'),'r') as f:
        reader=csv.reader(f)
        data=[]
        for ix,row in enumerate(reader):
            if ix==0:
                print(list(row))
                continue
            list_data=list(row)
            image_path=op.join(images_root,list_data[0])
            category=list_data[1]
            keypoints=list_data[2:]
            image=PI.open(image_path)
            item={}
            item['dataset']='FashionAI'
            item['img_paths']=image_path
            item['img_height']=image.size[1]
            item['img_width']=image.size[0]
            item['isValidation']=0
            item['numOtherPeople']=0
            item['people_index']=0
            item['annolist_index']=0
            ltx=1e4;lty=1e4;rbx=0;rby=0

            joint_self=[]   #24*3
            for kp_str in keypoints:
                keypoint=kp_str.split('_')
                x=int(keypoint[0])
                y=int(keypoint[1])
                visible=int(keypoint[2])
                if visible==-1:
                    visible=2
                joint=[x,y,visible]
                joint_self.append(joint)
                ltx=min(ltx,x)
                lty=min(lty,y)
                rbx=max(rbx,x)
                rby=max(rby,y)
            center=[0.5*(ltx+rbx),0.5*(lty+rby)]
            item['objpos']=center
            item['joint_self']=np.asarray(joint_self).transpose().tolist()   #3*24
            item['scale_provided']=1.0*item['img_height']/net_input_side#align height
            data.append(item)
    label=dict(root=data)
    with open('label_data.json','w') as f:
        f.write(json.dumps(label))
    print('done')


if __name__=='__main__':
    csv2json()
                
   
