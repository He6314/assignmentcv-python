#Functions for corner detection and basic similarity matching
#by Qin He

import numpy as np
import cv2 as cv
import math

def harrisCornerHQ(imgSrc, sigma):
	img = np.float32(imgSrc);
	pyrScale = 1;
	while img.shape[0]>1200 or img.shape[1]>1200:
		img = cv.pyrDown(img);
		pyrScale *= 2;
		print(img.shape)
	ftSz = int(4*sigma)+1;
	if(ftSz%2!=0):
		ftSz+=1;
	img = cv.GaussianBlur(img,(5,5),1);

	harrisFilterX = np.ones([ftSz,ftSz],np.float32);
	harrisFilterX[:,0:ftSz/2] *= -1;
	harrisFilterY = np.ones([ftSz,ftSz],np.float32);
	harrisFilterY[ftSz/2:ftSz,:] *= -1;

	dx = cv.filter2D(np.float32(img),-1,harrisFilterX);
	dy = cv.filter2D(np.float32(img),-1,harrisFilterY);

	dx2 = dx*dx;
	dy2 = dy*dy;
	dxdy = dx*dy;

	cnSz = int(5*sigma)+1;
	if(cnSz%2==0):
		cnSz+=1;
	sePt = (cnSz-1)/2;#start & end point

	valueMap = np.zeros(img.shape,np.float32);
	for i in range(sePt,img.shape[0]-sePt-1):
		for j in range(sePt,img.shape[1]-sePt-1):
			patchX = dx2[i-sePt:i+sePt+1,j-sePt:j+sePt+1];
			patchY = dy2[i-sePt:i+sePt+1,j-sePt:j+sePt+1];
			patchXY = dxdy[i-sePt:i+sePt+1,j-sePt:j+sePt+1];
			C = np.matrix([[patchX.sum(),patchXY.sum()],[patchXY.sum(),patchY.sum()]]);
			if(C.trace()==0):
				valueMap[i,j] = 0;
			else:
				valueMap[i,j] = np.linalg.det(C)/(C.trace()*C.trace());

	cornerMap = np.zeros(imgSrc.shape,np.int);
	resultList = [];	
	threshold = valueMap.mean();#0.0826;#=> 0.1/(1+0.1)^2, it works very well for simple images

	sePt = 15;
	for i in range(sePt,img.shape[0]-sePt-1):
		for j in range(sePt,img.shape[1]-sePt-1):
			patchValue = valueMap[i-sePt:i+sePt+1,j-sePt:j+sePt+1];
			#patchMask = patchValue>=threshold;
			value = valueMap[i,j];
			if (value>threshold) and (value==patchValue.max()): 
			#if patchMask.min() and and (value==patchValue.max()):, it works very well for simple images
				cornerMap[i*pyrScale,j*pyrScale] = 1;
				resultList.append([i*pyrScale,j*pyrScale]);
	print('HARRIS CORNER DETECTION COMPLETE')
	return resultList, cornerMap, valueMap;


def ssdHQ(list1,img1,list2,img2):
	N = (img1.shape[0]+img1.shape[1]+img2.shape[0]+img2.shape[1])/4/25;
	if(N%2==0):
		N = N+1;
	sePt = (N-1)/2;
	
	img1 = cv.copyMakeBorder(img1,sePt,sePt,sePt,sePt,cv.BORDER_CONSTANT,value=0);
	img2 = cv.copyMakeBorder(img2,sePt,sePt,sePt,sePt,cv.BORDER_CONSTANT,value=0);

	ssdList = [];
	drawFlag = True;
	for i in range(len(list1)):
		patch1 = img1[list1[i][0]:list1[i][0]+2*sePt+1,list1[i][1]:list1[i][1]+2*sePt+1];
		for j in range(len(list2)):
			patch2 = img2[list2[j][0]:list2[j][0]+2*sePt+1,list2[j][1]:list2[j][1]+2*sePt+1];
			diff = patch1-patch2;
			diff2 = diff*diff;
			ssdList.append([i,j,diff2.sum()]);
	print('SSD LISTED')
	return ssdList;


def nccHQ(list1,img1,list2,img2):
	N = (img1.shape[0]+img1.shape[1]+img2.shape[0]+img2.shape[1])/4/25;
	if(N%2==0):
		N = N+1;
	sePt = (N-1)/2;
	
	img1 = cv.copyMakeBorder(img1,sePt,sePt,sePt,sePt,cv.BORDER_CONSTANT,value=0);
	img2 = cv.copyMakeBorder(img2,sePt,sePt,sePt,sePt,cv.BORDER_CONSTANT,value=0);
	
	nccList = [];
	for i in range(len(list1)):
		patch1 = np.float64(img1[list1[i][0]:list1[i][0]+2*sePt+1,list1[i][1]:list1[i][1]+2*sePt+1]);
		m1 = patch1.mean();
		diff1 = patch1-m1;
		diff1S = diff1*diff1;
		for j in range(len(list2)):
			patch2 = np.float64(img2[list2[j][0]:list2[j][0]+2*sePt+1,list2[j][1]:list2[j][1]+2*sePt+1]);
			m2 = patch2.mean();
			diff2 = patch2-m2;
			diff2S = diff2*diff2;
			
			diff1x2 = diff1*diff2;
			if(diff1S.sum()*diff2S.sum()==0):
				nccValue = 0;
			else:
				nccValue = diff1x2.sum()/math.sqrt(diff1S.sum()*diff2S.sum());
			nccList.append([i,j,nccValue]);
	print('NCC LISTED')
	return nccList;


def drawLineHQ(corrList,img1,img2, color, map1,map2):
	result = np.zeros([img1.shape[0],img1.shape[1]+img2.shape[1]+50,img1.shape[2]]);

	img1[map1==1] = [0,0,255];
	img2[map2==1] = [0,0,255];

	result[0:img1.shape[0],0:img1.shape[1],:] = img1;
	result[0:img2.shape[0],img1.shape[1]+50:result.shape[1],:] = img2;
	
	for i in range(len(corrList)):
		coord1 = list(corrList[i][0]);
		coord2 = list(corrList[i][1]);
		coord2[1] += img1.shape[1]+50-1;
		coord1.reverse();
		coord2.reverse();
		result = cv.line(result,tuple(coord1),tuple(coord2),color);	
	print('MATCH COMPLETE')
	return result;


