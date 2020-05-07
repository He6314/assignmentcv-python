#Functions for Otsu's segmentation
#by Qin He

import numpy as np
import math

def img2hist(singleChannel):
	width = singleChannel.shape[0];
	height = singleChannel.shape[1];
	pixelAmount = width*height;

	upper = int(singleChannel.max());
	lower = int(singleChannel.min());
	hist = np.zeros([1,upper+1-lower]);
	for i in range(width):
		for j in range(height):
			binNb = int(singleChannel[i,j])-lower;
			if(binNb>=0):
				hist[0,binNb] += 1;
	#hist /= pixelAmount;
	hist = np.int64(hist);
	return hist,range(lower,upper+1);


def minBitVector(rawPattern):
	pattern = rawPattern[:];
	arrays = [];
	values = [];
	for i in range(len(pattern)):
		value = 0;
		array = pattern[1:len(pattern)];
		array.append(pattern[0]);
		for j in range(1,len(array)+1):		
			value += 2**(j)*array[-j];
		arrays.append(array);
		values.append(value);
		pattern = array[:];
	del pattern;
	return arrays[values.index(min(values))];	


def encodePattern(minPattern):
	pattern = minPattern[:];
	n = 0;
	for i in range(len(pattern)-1):
		if(pattern[i+1]!=pattern[i]):
			n+=1;
	if(n>2):
		return len(minPattern)+1;
	elif(n==0) and (minPattern[0]==1):
		return len(minPattern);
	elif(n==0) and (minPattern[0]==0):
		return 0;
	else:
		return sum(minPattern);


def calBitPattern(img, i,j, P, R):
	rawPattern = [];
	for p in range(P):
		x = R * math.cos(2*math.pi*p/P);
		y = R * math.sin(2*math.pi*p/P);
		if abs(x) < 0.001: x = 0.0;
		if abs(y) < 0.001: y = 0.0;
		
		k,l = i+x, j+y;
		k_base,l_base = int(k),int(l);
		delta_k,delta_l = k-k_base,l-l_base;

		if (delta_k < 0.001) and (delta_l < 0.001):
			neighbor_value = float(img[k_base][l_base]);
		elif(delta_l < 0.001):
			neighbor_value = (1-delta_k) * img[k_base][l_base] + delta_k * img[k_base+1][l_base];
		elif (delta_k < 0.001):
			neighbor_value = (1-delta_l) * img[k_base][l_base] + delta_l * img[k_base][l_base+1];
		else:
			neighbor_value = (1-delta_k)*(1-delta_l) * img[k_base][l_base] + \
                                         (1-delta_k)* delta_l    * img[k_base][l_base+1]  + \
                                            delta_k * delta_l    * img[k_base+1][l_base+1]  + \
                                            delta_k *(1-delta_l) * img[k_base+1][l_base];

		if (neighbor_value >= img[i][j]):
			rawPattern.append(1);
		else:
			rawPattern.append(0);	
	minPattern = minBitVector(rawPattern);
	return encodePattern(minPattern);

def lbpFeatures(img, P=8, R=1):
	features = np.zeros(img.shape);
	for i in range(int(R),img.shape[0]-int(R)):
		for j in range(int(R),img.shape[1]-int(R)):
			features[i][j] = calBitPattern(img, i,j, P, R);
	hist, xcoord = img2hist(features);
	return features, hist, xcoord;

def kNN(test, trainSet, k=5):
	nbClasses = trainSet.shape[0];
	nbTrainImg = trainSet.shape[1];
	lenFeatures = trainSet.shape[2];
	
	distMap = np.ones([nbClasses,nbTrainImg]);
	distMap *= (test.max()*2)**2;
	for i in range(nbClasses):
		for j in range(nbTrainImg):
			train = trainSet[i][j];
			distMap[i][j] = np.linalg.norm(test-train);
	
	nearst = np.ones([k,2]);
	nearst *= (test.max()*2)**2;
	for i in range(nbClasses):
		for j in range(nbTrainImg):
			for m in range(k):
				if(distMap[i][j]<nearst[m][0]):
					for n in range(k-1,m,-1):
						nearst[n] = nearst[n-1];
					nearst[m][0] = distMap[i][j];
					nearst[m][1] = i;
					break;
	freq = np.bincount(np.int64(nearst[:,1]));
	return np.argmax(freq);


