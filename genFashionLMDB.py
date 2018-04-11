# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 16:07:18 2018

@author: samsung
"""
"""
In Python2, the result of struct.pack is a string.
To get the value, ord is needed: v=ord(b[0])

In Python3, the result of struct.pack is a byte type. 
No ord is needed, the value can be accessed directly by index: v=b[0]
"""
import lmdb
import os
import cv2
import json
import os.path as op
import struct
import sys
sys.path.insert(0,'/data/xiaobing.wang/pingjun.li/yfji/Realtime_Multi-Person_Pose_Estimation-master/caffe_train-master/python')
import caffe
import numpy as np

def writeLMDB(dataset, lmdb_path, validation=0):
    env = lmdb.open(lmdb_path, map_size=int(1e12))
    txn = env.begin(write=True)
    with open('label_data.json','r') as f:
        label_data=json.load(f)
    data=label_data['root']
    numSamples=len(data)
    random_order = np.random.permutation(numSamples).tolist()

    validArray=[int(d['isValidation']) for d in data]
    writeCount=0
    totalWriteCount=validArray.count(0)
    
    for count in range(numSamples):
        idx=random_order[count]
        img = cv2.imread(data[idx]['img_paths'])
        height = img.shape[0]
        width = img.shape[1]    
        meta_data = np.zeros(shape=(height,width,1), dtype=np.uint8)
        clidx = 0 # current line index
        for i in range(len(data[idx]['dataset'])):
            meta_data[clidx][i] = ord(data[idx]['dataset'][i])
        clidx = clidx + 1
        # image height, image width
        height_binary = float2bytes(data[idx]['img_height'])
        for i in range(len(height_binary)):
            meta_data[clidx][i] = ord(height_binary[i])
        width_binary = float2bytes(data[idx]['img_width'])
        for i in range(len(width_binary)):
            meta_data[clidx][4+i] = ord(width_binary[i])
        clidx = clidx + 1
        # (a) isValidation(uint8), numOtherPeople (uint8), people_index (uint8), annolist_index (float), writeCount(float), totalWriteCount(float)
        meta_data[clidx][0] = data[idx]['isValidation']
        meta_data[clidx][1] = data[idx]['numOtherPeople']
        meta_data[clidx][2] = data[idx]['people_index']
        annolist_index_binary = float2bytes(data[idx]['annotlist_index'])
        for i in range(len(annolist_index_binary)): # 3,4,5,6
            meta_data[clidx][3+i] = ord(annolist_index_binary[i])
        count_binary = float2bytes(float(writeCount)) # note it's writecount instead of count!
        for i in range(len(count_binary)):
            meta_data[clidx][7+i] = ord(count_binary[i])
        totalWriteCount_binary = float2bytes(float(totalWriteCount))
        for i in range(len(totalWriteCount_binary)):
            meta_data[clidx][11+i] = ord(totalWriteCount_binary[i])
        nop = int(data[idx]['numOtherPeople'])
        clidx = clidx + 1
        # (b) objpos_x (float), objpos_y (float)
        objpos_binary = float2bytes(data[idx]['objpos'])
        for i in range(len(objpos_binary)):
            meta_data[clidx][i] = ord(objpos_binary[i])
        clidx = clidx + 1
        # (c) scale_provided (float)
        scale_provided_binary = float2bytes(data[idx]['scale_provided'])
        for i in range(len(scale_provided_binary)):
            meta_data[clidx][i] = ord(scale_provided_binary[i])
        clidx = clidx + 1
        # (d) joint_self (3*16) (float) (3 line)
        joints = np.asarray(data[idx]['joint_self']).tolist() # transpose to 3*16
        for i in range(len(joints)):
            row_binary = float2bytes(joints[i])
            for j in range(len(row_binary)):
                meta_data[clidx][j] = ord(row_binary[j])
            clidx = clidx + 1
        img4ch = np.concatenate((img, meta_data), axis=2)
        img4ch = np.transpose(img4ch, (2, 0, 1))
        print(img4ch.shape)
		
        datum = caffe.io.array_to_datum(img4ch, label=0)
        key = '%07d' % writeCount
        txn.put(key, datum.SerializeToString())
        if(writeCount % 1000 == 0):
            txn.commit()
            txn = env.begin(write=True)
        print('%d/%d/%d/%d' % (count,writeCount,idx,numSamples))
        writeCount = writeCount + 1
    txn.commit()
    env.close()
    print('done')
          

def float2bytes(floats):
	if type(floats) is float:
		floats = [floats]
	if type(floats) is int:
		floats=[float(floats)]
	return struct.pack('%sf' % len(floats), *floats)
 
if __name__=='__main__':
#    print(float2bytes([1.1,2.2]))
	writeLMDB('FashionAI','../lmdb')
