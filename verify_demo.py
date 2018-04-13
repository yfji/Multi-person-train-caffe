# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:38:53 2018

@author: samsung
"""

import cv2
import numpy as np
import os

feature_maps=np.zeros((18,452,742),dtype=np.uint8)
feature_maps[0][96,39]=1
feature_maps[0][80,313]=1
feature_maps[0][86,418]=1
feature_maps[1][139,48]=1
feature_maps[1][116,313]=1
feature_maps[1][138,436]=1
feature_maps[2][155,6]=1
feature_maps[2][119,275]=1
feature_maps[2][181,391]=1
feature_maps[3][146,212]=1
feature_maps[3][234,477]=1
feature_maps[4][137,145]=1
feature_maps[4][255,576]=1
feature_maps[5][138,96]=1
feature_maps[5][122,358]=1
feature_maps[5][122,513]=1
feature_maps[6][81,138]=1
feature_maps[6][172,410]=1
feature_maps[6][90,559]=1
feature_maps[7][24,130]=1
feature_maps[7][239,433]=1
feature_maps[7][131,639]=1

feature_maps[8][278,55]=1
feature_maps[8][220,291]=1
feature_maps[8][250,447]=1
feature_maps[9][343,71]=1
feature_maps[9][291,207]=1  #
feature_maps[9][340,430]=1
feature_maps[10][426,132]=1
feature_maps[10][369,280]=1
feature_maps[10][417,405]=1
feature_maps[11][243,121]=1
feature_maps[11][220,346]=1
feature_maps[11][251,512]=1

feature_maps[12][291,207]=1 #
feature_maps[12][304,365]=1
feature_maps[12][313,575]=1
feature_maps[13][368,201]=1
feature_maps[13][394,378]=1
feature_maps[13][400,516]=1

pafs=[[[(139,48),(96,39)],[(116,313),(80,313)],[(138,436),(86,418)]],
      [[(139,48),(155,6)],[(116,313),(119,275)],[(138,436),(181,391)]],
      [[(139,48),(138,96)],[(116,313),(122,358)],[(138,436),(122,513)]],
      [[(119,275),(146,212)],[(181,391),(234,477)]],
      [[(146,212),(137,145)],[(234,477),(255,576)]],
      [[(138,96),(81,138)],[(122,358),(172,410)],[(122,513),(90,559)]],
      [[(81,138),(24,130)],[(172,410),(239,433)],[(90,559),(131,639)]],
      [[(139,48),(278,55)],[(116,313),(220,291)],[(138,436),(250,447)]],
      [[(278,55),(343,71)],[(220,291),(291,207)],[(250,447),(340,430)]],
      [[(343,71),(426,132)],[(291,207),(369,280)],[(340,430),(417,405)]],
      [[(139,48),(243,121)],[(116,313),(220,346)],[(138,436),(251,512)]],
      [[(243,121),(291,207)],[(220,346),(304,365)],[(251,512),(313,575)]],
      [[(291,207),(368,201)],[(304,365),(394,378)],[(313,575),(400,516)]]
        ]


        
limbSeq=[[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
         ]

peak_start=0
all_peaks=[]
for part in range(19-1):
    heatmap=feature_maps[part]  
    #left top to right bottom
    peaks=np.hstack((np.nonzero(heatmap)[0].reshape(-1,1), np.nonzero(heatmap)[1].reshape(-1,1)))  
#    print(peaks)
    peaks_with_score=[(peak[0],peak[1])+(heatmap[peak[0],peak[1]],) for peak in peaks]
    peaks_id=range(peak_start,peak_start+peaks.shape[0])                  
    peaks_with_score_and_id=[peaks_with_score[i]+(peaks_id[i],) for i in range(len(peaks_id))]
    peak_start+=peaks.shape[0]
    all_peaks.append(peaks_with_score_and_id)

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
                pair=[locA, locB]
                for points in pafs[k]:
                    if locA[0]==points[0][0] and locA[1]==points[0][1] and locB[0]==points[1][0] and locB[1]==points[1][1]:
                        print('found')
                        connection_candidate.append([i,j])
    
    connection=np.zeros((0,4))
    for c in range(len(connection_candidate)):
        i,j=connection_candidate[c][0:2]
        if i not in connection[:2] and j not in connection[:3]:
            connection=np.vstack((connection, np.asarray([candA[i][3],candB[j][3], i,j])))
            if len(connection)>min(nA,nB):
                break
    connection_all.append(connection)

print(connection_all)
subset = -1 * np.ones((0, 20))
candidate = np.array([item for sublist in all_peaks for item in sublist])   #[i,j,s,id,i,j,s,id...]

for k in range(len(pafs)):
    partA_ids=connection_all[k][:,0]
    partB_ids=connection_all[k][:,1]    
    indexA,indexB=np.asarray(limbSeq[k])-1
    
    for i in range(len(connection_all[k])): #all limbs
        found=0
        sub_idx=[-1,-1]
        for j in range(len(subset)):
            if subset[j][indexA]==partA_ids[i] or subset[j][indexB]==partB_ids[i]:
                sub_idx[found]=j
                found+=1
        if found==1:
            print('found=1')
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
                subset[j1][:-2] += (subset[j2][:-2] + 1)
                subset[j1][-2:] += subset[j2][-2:]
                subset[j1][-2] += connection_all[k][i][2]
                subset = np.delete(subset, j2, 0)
            else: # as like found == 1	#shoulder is shared (co-source)
                subset[j1][indexB] = partB_ids[i]
                subset[j1][-1] += 1
                subset[j1][-2] += candidate[partB_ids[i].astype(int), 2] + connection_all[k][i][2]

        elif not found and k < 17:
            row = -1 * np.ones(20)
            row[indexA] = partA_ids[i]	#one row is a person. person[index] is the index of keypoints
            row[indexB] = partB_ids[i]
            row[-1] = 2
            row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
            subset = np.vstack([subset, row])
