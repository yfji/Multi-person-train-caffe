# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:08:46 2018

@author: samsung
"""

import caffe
import numpy as np
import yaml
import json
import cv2

class CPMDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self.label_path_= layer_params['label_path']
        self.batch_size_=layer_params['batch_size']
        self.stride_=layer_params['stride']
        self.visualize_=layer_params['visualize']
        self.height_=layer_params['height']
        self.width_=layer_params['width']
        self.upper_body_=layer_params['upper_body']
        self.dataset_=layer_params['dataset']
        self.cur_index_=0
        self.num_parts_=0
        self.np_ours_=0
        self.np_plus_paf_=0
        self.slice_points_=None
        
        with open(self.label_path, 'r') as f:
            label_json=json.load(f)
        self.data=label_json['root']
        self.random_order=np.random.permutation(range(len(data))).tolist()
        
    def forward(self, bottom, top):
        label_chn=self.calc_label_channels()
        blob_data=np.zeros((self.batch_size_, 3, self.height_, self.width_), dtype=np.float32)
        blob_label=np.zeros((self.batch_size_, label_chn, self.height_/self.stride_, self.width_/self.stride_))
        for i in range(self.batch_size_):
            blob_data[i]=self.load_data_channel(i)
            
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass
    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
    
    def load_data_channel(self, index):
        record=self.data[self.random_order[index]]
        img_path=record['img_paths']        
        img_height=record['img_height']
        img_width=record['img_width']
        image=cv2.imread(img_path)
        if img_height!=image.shape[0] or img_width!=image.shape[1]:
            raise Exception('Image shape not unified')
        objpos=record
        
    def calc_label_channels(self):
        channel=0
        if self.dataset_=='COCO' and self.upper_body_:
            self.num_parts_=11
            self.np_ours_=12
            self.np_plus_paf_=38
        elif self.dataset_=='COCO':
            self.num_parts_=17
            self.np_ours_=18
            self.np_plus_paf_=56
        elif self.dataset_=='MPI':
            self.num_parts_=15
            self.np_ours_=15
            self.np_plus_paf_=43
        elif self.dataset_=='HAND':
            self.num_parts_=21
            self.np_ours_=21
            self.np_plus_paf_=21
        elif self.dataset_=='FashionAI':
            self.num_parts_=24
            self.np_ours_=24
            self.np_plus_paf_=24
        elif self.dataset_=='FOOTTOP':
            self.num_parts_=19
            self.np_ours_=20
            self.np_plus_paf_=62
        else:
            raise Exception('Unknown dataset type')
        channel=2*(self.np_plus_paf_+1)
        return channel
            