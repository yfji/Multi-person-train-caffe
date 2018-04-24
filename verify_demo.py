# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:38:53 2018

@author: samsung
"""

import cv2
import numpy as np
import os

feature_maps=np.zeros((14,452,742),dtype=np.uint8)
feature_maps[0,96,39]=1
feature_maps[0,80,313]=1
feature_maps[0,86,418]=1
feature_maps[1,139,48]=1
feature_maps[1,116,313]=1
feature_maps[1,138,436]=1
feature_maps[2,155,6]=1
feature_maps[2,119,275]=1
feature_maps[2,181,391]=1
feature_maps[3,146,212]=1
feature_maps[3,234,477]=1
feature_maps[4,137,145]=1
feature_maps[4,255,576]=1
feature_maps[5,138,96]=1
feature_maps[5,122,358]=1
feature_maps[5,122,513]=1
feature_maps[6,81,138]=1
feature_maps[6,172,410]=1
feature_maps[6,90,559]=1
feature_maps[7,24,130]=1
feature_maps[7,239,433]=1
feature_maps[7,131,639]=1

feature_maps[8,278,55]=1
feature_maps[8,220,291]=1
feature_maps[8,250,447]=1
feature_maps[9,343,71]=1
#feature_maps[9,299,260]=1  #
feature_maps[9,340,430]=1
feature_maps[9,340,430]=1
feature_maps[10,426,132]=1
feature_maps[10,369,280]=1
feature_maps[10,417,405]=1
feature_maps[11,243,121]=1
feature_maps[11,220,346]=1
feature_maps[11,251,512]=1

feature_maps[12,291,207]=1 #
feature_maps[12,304,365]=1
feature_maps[12,313,575]=1
feature_maps[13,368,201]=1
feature_maps[13,394,378]=1
feature_maps[13,400,516]=1

pafs=[[[(139,48),(96,39)],[(116,313),(80,313)],[(138,436),(86,418)]],
      [[(139,48),(155,6)],[(116,313),(119,275)],[(138,436),(181,391)]],#1,2
      [[(139,48),(138,96)],[(116,313),(122,358)],[(138,436),(122,513)]],
      [[(119,275),(146,212)],[(181,391),(234,477)]],
      [[(146,212),(137,145)],[(234,477),(255,576)]],
      [[(138,96),(81,138)],[(122,358),(172,410)],[(122,513),(90,559)]],
      [[(81,138),(24,130)],[(172,410),(239,433)],[(90,559),(131,639)]],
      [[(139,48),(278,55)],[(116,313),(220,291)],[(138,436),(250,447)]],
      [[(278,55),(343,71)],[(220,291),(340,430)],[(250,447),(340,430)]],#8,9
      [[(343,71),(426,132)],[(340,430),(369,280)],[(340,430),(417,405)]],#9,10
      [[(139,48),(243,121)],[(116,313),(220,346)],[(138,436),(251,512)]],#1,11
      [[(243,121),(291,207)],[(220,346),(304,365)],[(220,346),(313,575)]],#11,12
      [[(291,207),(368,201)],[(304,365),(394,378)],[(313,575),(400,516)]]#12,13
        ]

canvas=np.zeros((452,742,1),dtype=np.uint8)
for i in range(15-1):
    fm=feature_maps[i]
    ys=np.nonzero(fm)[0]
    xs=np.nonzero(fm)[1]
    assert(len(ys)==len(xs));
    for c in range(len(ys)):
        cv2.circle(canvas, (xs[c],ys[c]), 6, 255, -1)
for i in range(len(pafs)):
    fm_vecs=pafs[i]
    for j in range(len(fm_vecs)):
        vec=fm_vecs[j]
        start=vec[0]
        end=vec[1]
        cv2.line(canvas, (start[1],start[0]), (end[1],end[0]), 255, 2)
        
limbSeq=[[1,0],[1,2],
         [1,5],[2,3],
         [3,4],[5,6],
         [6,7],[1,8],
         [8,9],[9,10],
         [1,11],[11,12],
         [12,13],
         ]

peak_start=0
all_peaks=[]
for part in range(15-1):
    heatmap=feature_maps[part]  
    #left top to right bottom
    peaks=np.hstack((np.nonzero(heatmap)[0].reshape(-1,1), np.nonzero(heatmap)[1].reshape(-1,1)))  
#    print(peaks)
    peaks_with_score=[(peak[0],peak[1])+(heatmap[peak[0],peak[1]],) for peak in peaks]
    peaks_id=range(peak_start,peak_start+peaks.shape[0])                  
    peaks_with_score_and_id=[peaks_with_score[i]+(peaks_id[i],) for i in range(len(peaks_id))]
    peak_start+=peaks.shape[0]
    all_peaks.append(peaks_with_score_and_id)   #[[x,y,score,id],...]

connection_all=[]
for k in range(len(pafs)):
    candA=all_peaks[limbSeq[k][0]]
    candB=all_peaks[limbSeq[k][1]]
    nA=len(candA)
    nB=len(candB)
    connection_candidate=[]
    if nA!=0 and nB!=0:
        for i in range(nA):
            locA=(candA[i][0],candA[i][1])
            for j in range(nB):
                locB=(candB[j][0],candB[j][1])
                for points in pafs[k]:
                    if locA[0]==points[0][0] and locA[1]==points[0][1] and locB[0]==points[1][0] and locB[1]==points[1][1]:
                        print('found (%d,%d)'%(candA[i][3],candB[j][3]))
                        connection_candidate.append([i,j])
    print(connection_candidate)
    connection=np.zeros((0,4))
    for c in range(len(connection_candidate)):
        i,j=connection_candidate[c][0:2]
        if i not in connection[:,2] and j not in connection[:,3]:
            connection=np.vstack((connection, np.asarray([candA[i][3],candB[j][3], i,j])))
            if len(connection)>min(nA,nB):
                break
        else:
            print('repeated')
            print(connection[:,2])
            print(connection_candidate[c])
    connection_all.append(connection)
    
print(all_peaks)
print(connection_all)
subset = -1 * np.ones((0, 20))
candidate = np.array([item for sublist in all_peaks for item in sublist])   #[i,j,s,id,i,j,s,id...]


m_canvas=np.zeros((452,742,3),dtype=np.uint8)
xs=np.nonzero(canvas==255)[1]
ys=np.nonzero(canvas==255)[0]
m_canvas[ys,xs,:]=255

for k in range(len(pafs)):
    partA_ids=connection_all[k][:,0]
    partB_ids=connection_all[k][:,1]    
    indexA,indexB=np.asarray(limbSeq[k])
    
    for i in range(len(connection_all[k])): #all limbs
        found=0
        sub_idx=[-1,-1]
        idA=0
        idB=0
        for j in range(len(subset)):
            if subset[j][indexA]==partA_ids[i] or subset[j][indexB]==partB_ids[i]:
                sub_idx[found]=j
                idA=partA_ids[i]
                idB=partB_ids[i]
                found+=1
        if found==1:
            print('limb (%d,%d) found=1'%(indexA, indexB))
            j = sub_idx[0]
            if(subset[j][indexB] != partB_ids[i]):
                subset[j][indexB] = partB_ids[i]
                subset[j][-1] += 1
                subset[j][-2] += candidate[partB_ids[i].astype(int), 2] + connection_all[k][i][2]
        elif found == 2: # if found 2 and disjoint, merge them
            j1, j2 = sub_idx	#person j1, person j2
            print("found = 2")
            membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
            if len(np.nonzero(membership == 2)[0]) == 0: #merge	#disjoint	#elbow is shared (co-terminal)
                print('co-terminal')
                subset[j1][:-2] += (subset[j2][:-2] + 1)
                subset[j1][-2:] += subset[j2][-2:]
                subset[j1][-2] += connection_all[k][i][2]
                subset = np.delete(subset, j2, 0)
            else: # as like found == 1	#shoulder is shared (co-source)
                print('co-source: (%d,%d)'%(indexA, indexB))
                subset[j1][indexB] = partB_ids[i]
                subset[j1][-1] += 1
                subset[j1][-2] += candidate[partB_ids[i].astype(int), 2] + connection_all[k][i][2]

        elif not found and k < 17:
            print('limb (%d,%d) create new'%(indexA, indexB))
            row = -1 * np.ones(20)
            row[indexA] = partA_ids[i]	#one row is a person. person[index] is the index of keypoints
            row[indexB] = partB_ids[i]
            idA=partA_ids[i]
            idB=partB_ids[i]
            row[-1] = 2
            row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
            subset = np.vstack([subset, row])
            
        for h in range(len(all_peaks[indexA])):
            if all_peaks[indexA][h][-1]==idA:
                for n in range(len(all_peaks[indexB])):
                    if all_peaks[indexB][n][-1]==idB:
                        pt1=(int(all_peaks[indexA][h][1]),int(all_peaks[indexA][h][0]))
                        pt2=(int(all_peaks[indexB][n][1]),int(all_peaks[indexB][n][0]))
                        cv2.circle(m_canvas, pt1, 4, (0,0,255),-1)
                        cv2.circle(m_canvas, pt2, 4, (0,255,0),-1)
                        cv2.line(m_canvas,pt1,pt2,(0,0,255),2)
                        break
                break
        cv2.imshow('steps',m_canvas)
        cv2.waitKey()

cv2.imshow('canvas', canvas)
cv2.waitKey()
cv2.destroyAllWindows()
