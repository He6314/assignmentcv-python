#Functions for simple wide-angle images synthesis
#by Qin He

import numpy as np
import cv2 as cv
import math
import random as rd

def findHomographyHQ(coords_range, coords_domain):
	A = np.zeros((2*len(coords_range),9));
	H = np.zeros((3,3));
		
	for i in range(len(coords_range)):
		A[i*2] = [0,0,0,-coords_range[i][0],-coords_range[i][1],-1, coords_range[i][0]*coords_domain[i][1], coords_range[i][1]*coords_domain[i][1], coords_domain[i][1]];
		A[i*2+1] = [coords_range[i][0],coords_range[i][1],1,0,0,0,(-1*coords_range[i][0]*coords_domain[i][0]),(-1*coords_range[i][1]*coords_domain[i][0]), -coords_domain[i][0]];
	
	U,Dsq,V = np.linalg.svd(A);
	vecH = V[V.shape[0]-1];
	for i in range(vecH.size):
		H[i/3,i%3] = vecH[i];
	return H;

def outlierRejectionHQ(srcPts1, srcPts2, chosenPts1, chosenPts2, threshold):
	H = findHomographyHQ(chosenPts1, chosenPts2);
	inliers1 = [];
	inliers2 = [];
	allPts1 = srcPts1[:];
	allPts2 = srcPts2[:];
	distS = [];
	for i in range(len(allPts1)):
		x = np.array([allPts1[i][0], allPts1[i][1], 1]);
		xp= np.array([allPts2[i][0], allPts2[i][1], 1]);
		Hx = np.dot(H,x);
		Hx /= Hx[2];
		dist = np.linalg.norm(Hx - xp);
		distS.append(dist);
		if(dist<threshold):
			inliers1.append(allPts1[i]);
			inliers2.append(allPts2[i]);
	distS = np.array(distS);
	#print(len(inliers1))
	#print("outlier rejected");
	return H, inliers1, inliers2;


def ransacHQ(interestPoints1,interestPoints2, eps, p, n, delta):
	M = int(round(len(interestPoints1)*(1-eps)));
	N = int(round(math.log(1-p)/(math.log(1-(pow((1-eps),n))))));
	print("M",M,"N",N);

	outPts1 = range(M);
	outPts2 = range(M);
	Hout = np.array([[1,0,0],[0,1,0],[0,0,1]]);
	for i in range(N):
		leftPts1 = interestPoints1[:];
		leftPts2 = interestPoints2[:];
		chosenPts1 = [];
		chosenPts2 = [];
		for j in range(n):
			index = rd.randint(0,len(leftPts1)-1);
			chosenPts1.append(leftPts1[index]);
			del leftPts1[index];
			chosenPts2.append(leftPts2[index]);
			del leftPts2[index];

		H, inliers1, inliers2 = outlierRejectionHQ(interestPoints1,interestPoints2,chosenPts1,chosenPts2,delta);

		if(len(inliers1)>M and len(inliers1)>len(outPts1)):
			outPts1 = inliers1[:];
			outPts2 = inliers2[:];
			Hout = H;

	return Hout, outPts1, outPts2;

def AuHoCaHQ(img1,img2,eps,p,n,delta,name = None):
	imgRGB1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB);
	imgRGB2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB);

	siftDetector = cv.xfeatures2d.SIFT_create(1000);
	pointList1, features1 = siftDetector.detectAndCompute(imgRGB1,None);
	pointList2, features2 = siftDetector.detectAndCompute(imgRGB2,None);
	bf = cv.BFMatcher(cv.NORM_L2,crossCheck=True);
	matches12 = bf.match(features1,features2);
	print("matched")

	interestPoints1 = [];
	interestPoints2 = [];

	T1 = np.array([[1,0,-float(img1.shape[0])/2.0],[0,1,-float(img1.shape[1])/2.0],[0,0,1]]);
	T2 = np.array([[1,0,-float(img2.shape[0])/2.0],[0,1,-float(img2.shape[1])/2.0],[0,0,1]]);

	for i in range(len(matches12)):
		point1 = matches12[i].queryIdx;
		pt1 = np.array([pointList1[point1].pt[1],pointList1[point1].pt[0],1]);
		coord1 = np.dot(T1,pt1);
		interestPoints1.append([coord1[0]/coord1[2],coord1[1]/coord1[2],i]);
		point2 = matches12[i].trainIdx;
		pt2 = np.array([pointList2[point2].pt[1],pointList2[point2].pt[0],1]);
		coord2 = np.dot(T2,pt2);
		interestPoints2.append([coord2[0]/coord2[2],coord2[1]/coord2[2],i]);
	print("points arranged")

	H, in1, in2 = ransacHQ(interestPoints1,interestPoints2, eps, p, n, delta);
	print("homography calculated")

	if(name!=None):
		inList = [];
		inMatches = [];
		outMatches = matches12;
		for i in range(len(in1)):
			inList.append(in1[i][2]);
			inMatches.append(matches12[in1[i][2]]);
		for i in sorted(inList, reverse=True):
			del outMatches[i];
		imgIn = cv.drawMatches(img1,pointList1,img2,pointList2,inMatches,None,flags=2);
		imgOut = cv.drawMatches(img1,pointList1,img2,pointList2,outMatches,None,flags=2);
		cv.imwrite('inliers'+name+'.jpg',imgIn);
		cv.imwrite('outliers'+name+'.jpg',imgOut);

	homo = np.dot(np.dot(np.matrix(T2).I,H),T1);
	homo = np.array(homo);
	print(homo)
	print('')
	return homo;

def mosaic2(imageO,imageM,homography):
	#map the image to a proper canvas using given homography
	canvas1 = np.array([[0,0,1],[0,imageO.shape[1]-1,1],[imageO.shape[0]-1,imageO.shape[1]-1,1],[imageO.shape[0]-1,0,1]]);
	canvas1 = canvas1.T;
	canvas2 = np.array([[0,0,1],[0,imageM.shape[1]-1,1],[imageM.shape[0]-1,imageM.shape[1]-1,1],[imageM.shape[0]-1,0,1]]);
	canvas2 = np.dot(np.matrix(homography).I,canvas2.T);

	canvas2[0] = canvas2[0]/canvas2[2];
	canvas2[1] = canvas2[1]/canvas2[2];
	canvas2[2] = canvas2[2]/canvas2[2];

	print(canvas1)
	print(canvas2)	

	wldH = int(max(canvas1[0].max(),canvas2[0].max()))+1;
	wldW = int(max(canvas1[1].max(),canvas2[1].max()))+1;

	wldO1 = int(min(canvas1[0].min(),canvas2[0].min()))-1;
	wldO2 = int(min(canvas1[1].min(),canvas2[1].min()))-1;

	wldOffset = [0-wldO1,0-wldO2];

	canvasWld = [wldH-wldO1,wldW-wldO2];
	print('')
	print('----------------------------')
	print([wldH,wldW])
	print([wldO1,wldO2])
	print(canvasWld)
	print('----------------------------')
	#print(homography)

	result = np.zeros([canvasWld[0],canvasWld[1],3]);
	
	for i in range(int(canvas1[0].min()),int(canvas1[0].max())):
		for j in range(int(canvas1[1].min()),int(canvas1[1].max())):
			result[i+wldOffset[0],j+wldOffset[1]] = imageO[i,j];

	for i in range(int(canvas2[0].min()),int(canvas2[0].max())):
		for j in range(int(canvas2[1].min()),int(canvas2[1].max())):
			wld_coord = np.array([i,j,1]);
			img_coord = np.dot(homography,wld_coord);
			imgX = int(img_coord[0]/img_coord[2]);
			imgY = int(img_coord[1]/img_coord[2]);
			if(imgX>=0 and imgX<imageM.shape[0] and imgY>=0 and imgY<imageM.shape[1]):
				result[i+wldOffset[0],j+wldOffset[1]] = imageM[imgX,imgY];
			else:
				continue;
	return result;

def mosaic(images,Hs):
	#map the image to a proper canvas using given homography
	num_images = len(images);
	mid_point = (len(images)-1)/2;
	
	H = [];
	canvas = [];

	heights = [];
	widths = [];
	oXs = [];
	oYs = [];

	for i in range(len(images)):
		h = np.array([[1,0,0],[0,1,0],[0,0,1]]);
		if(i>mid_point):
			for j in range(i-mid_point):
				h = np.dot(Hs[i-j-1],h);
		elif(i<mid_point):
			for j in range(mid_point-i):
				h = np.dot(Hs[i+j],h);
		H.append(h);

		c = np.array([[0,0,1],[0,images[i].shape[1]-1,1],[images[i].shape[0]-1,images[i].shape[1]-1,1],[images[i].shape[0]-1,0,1]]);
		c = np.dot(np.matrix(h).I,c.T);	
		c[0] = c[0]/c[2];
		c[1] = c[1]/c[2];
		c[2] = c[2]/c[2];
		canvas.append(c);

		heights.append(c[0].max());
		widths.append(c[1].max());	
		oXs.append(c[0].min());	
		oYs.append(c[1].min());

	heights = np.array(heights);
	widths = np.array(widths);
	oXs = np.array(oXs);
	oYs = np.array(oYs);

	wldH = int(heights.max())+1;
	wldW = int(widths.max())+1;
	wldO1 = int(oXs.min())-1;
	wldO2 = int(oYs.min())-1;

	wldOffset = [0-wldO1,0-wldO2];
	canvasWld = [wldH-wldO1,wldW-wldO2];

	print('');
	print('----------------------------');
	print('maxHeight',wldH,'maxWidth',wldW);
	print('origin:', [wldO1,wldO2]);
	print('size:', canvasWld);
	print('----------------------------');
	print('');

	result = np.zeros([canvasWld[0],canvasWld[1],3]);
	
	for i in range(int(canvas[mid_point][0].min()),int(canvas[mid_point][0].max())):
		for j in range(int(canvas[mid_point][1].min()),int(canvas[mid_point][1].max())):
			wld_coord = np.array([i,j,1]);
			img_coord = np.dot(H[mid_point],wld_coord);
			imgX = int(img_coord[0]/img_coord[2]);
			imgY = int(img_coord[1]/img_coord[2]);
			if(imgX>=0 and imgX<images[mid_point].shape[0] and imgY>=0 and imgY<images[mid_point].shape[1]):
				result[i+wldOffset[0],j+wldOffset[1]] = images[mid_point][imgX,imgY];
			else:
				continue;
	cv.imwrite('0.jpg',result)

	for n in range(1,mid_point+1):
		left = mid_point-n;
		right = mid_point+n;
		for i in range(int(canvas[left][0].min()),int(canvas[left][0].max())):
			for j in range(int(canvas[left][1].min()),int(canvas[left][1].max())):
				wld_coord = np.array([i,j,1]);
				img_coord = np.dot(H[left],wld_coord);
				imgX = int(img_coord[0]/img_coord[2]);
				imgY = int(img_coord[1]/img_coord[2]);
				if(imgX>=0 and imgX<images[left].shape[0] and imgY>=0 and imgY<images[left].shape[1]):
					result[i+wldOffset[0],j+wldOffset[1]] = images[left][imgX,imgY];
				else:
					continue;
		for i in range(int(canvas[right][0].min()),int(canvas[right][0].max())):
			for j in range(int(canvas[right][1].min()),int(canvas[right][1].max())):
				wld_coord = np.array([i,j,1]);
				img_coord = np.dot(H[right],wld_coord);
				imgX = int(img_coord[0]/img_coord[2]);
				imgY = int(img_coord[1]/img_coord[2]);
				if(imgX>=0 and imgX<images[right].shape[0] and imgY>=0 and imgY<images[right].shape[1]):
					result[i+wldOffset[0],j+wldOffset[1]] = images[right][imgX,imgY];
				else:
					continue;
		cv.imwrite(str(n)+'.jpg',result)
	return result;
