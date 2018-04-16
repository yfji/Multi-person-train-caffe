# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:59:18 2018

@author: samsung
"""

import os
import os.path as op
import json
import PIL.Image as PI
import numpy as np

net_input_side=256

def genHandJson():
    dataset_root='/data/xiaobing.wang/pingjun.li/yfji/hand_labels_synth/'
    sub_dirs=os.listdir(dataset_root)
    #sub_dirs=[op.join(dataset_root, 'synth2'), op.join(dataset_root, 'synth3')]
    sub_dirs=[op.join(dataset_root, d) for d in sub_dirs if os.path.isdir(d)]
    data=[]
    for d in sub_dirs:
        dir_path=op.join(dataset_root,d)
        files=os.listdir(dir_path)
        files=sorted(files, key=lambda x:x[:x.rfind('.')])
        img_files=[f for f in files if f[f.rfind('.')+1:]=='jpg']
        json_files=[f for f in files if f[f.rfind('.')+1:]=='json']
        
        if len(img_files)!=len(json_files):
            continue
        
        for i in range(len(img_files)):
            label_path=os.path.join(dir_path,json_files[i])
        
            with open(label_path,'r') as f:
                label=json.load(f)
            hand_pts=label['hand_pts']
            
            entry={}
            entry['dataset']='CMU'
            entry['isValidation']=0
            entry['img_paths']=op.join(dir_path,img_files[i])
            
            image=PI.open(entry['img_paths'])
            entry['crop_x']=0
            entry['crop_y']=0
            entry['crop']=0
        
            entry['img_height']=image.size[1]
            entry['img_width']=image.size[0]
            entry['numOtherPeople']=0
            entry['people_index']=0
            entry['annolist_index']=0
            
            ltx=1e4;lty=1e4;rbx=0;rby=0
            for pt in hand_pts: #21*3
                if int(pt[2])==0:
                    continue
                ltx=min(ltx,pt[0])
                lty=min(lty,pt[1])
                rbx=max(rbx,pt[0])
                rby=max(rby,pt[1])
            hand_w=rbx-ltx
            hand_h=rby-lty
            im_w=entry['img_width']
            im_h=entry['img_height']                  
            hand_center=[0.5*(ltx+rbx),0.5*(lty+rby)]
            
            crop_side=max(hand_w,hand_h)
            
            if 1.0*im_w/hand_w>4:
                entry['crop_x']=max(0,hand_center[0]-crop_side)
                entry['crop']=1
            if 1.0*im_h/hand_h>4:
                entry['crop_y']=max(0,hand_center[1]-crop_side)
                entry['crop']=1
            if entry['crop']==1:
                crop_h=min(im_w,im_h,crop_side*2)
                crop_w=min(im_w,im_h,crop_side*2)
                entry['img_height']=min(im_h-entry['crop_y'], crop_h)
                entry['img_width']=min(im_w-entry['crop_x'], crop_w)
            entry['scale_provided']=1.0*entry['img_height']/net_input_side

            entry['objpos']=[hand_center[0]-entry['crop_x'],hand_center[1]-entry['crop_y']]
            joint_self=np.asarray(hand_pts)
            joint_self[:,0]=np.maximum(0,joint_self[:,0]-entry['crop_x'])
            joint_self[:,1]=np.maximum(0,joint_self[:,1]-entry['crop_y'])
            joint_self[joint_self[:,2]==0,2]=2
            
            joint_self=joint_self.tolist() #21*3
            entry['joint_self']=joint_self
            data.append(entry)
    with open('hand_label_cropt.json','w') as f:
        label_data=dict(root=data)
        print(json.dumps(label_data))
    print('done')

if __name__=='__main__':
    genHandJson()
