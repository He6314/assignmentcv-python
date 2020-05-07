#Some functions for 3D Scene Reconstruction
#by Qin He

import numpy as np
import cv2 as cv
import math
import random as rd

def calSSD(p1,p2,imgOri1,imgOri2):
	#calculate the SSD between 2 specific points
	N = 25; #(img1.shape[0]+img1.shape[1]+img2.shape[0]+img2.shape[1])/4/25; #found that fixed window size performs better
	if(N%2==0):
		N = N+1;
	sePt = (N-1)/2;
	
	img1 = cv.cvtColor(np.uint8(imgOri1), cv.COLOR_BGR2GRAY);
	img2 = cv.cvtColor(np.uint8(imgOri2), cv.COLOR_BGR2GRAY);
	img1 = cv.copyMakeBorder(img1,sePt,sePt,sePt,sePt,cv.BORDER_CONSTANT,value=0);
	img2 = cv.copyMakeBorder(img2,sePt,sePt,sePt,sePt,cv.BORDER_CONSTANT,value=0);

	patch1 = np.float64(img1[p1[0]:p1[0]+2*sePt+1,p1[1]:p1[1]+2*sePt+1]);
	patch2 = np.float64(img2[p2[0]:p2[0]+2*sePt+1,p2[1]:p2[1]+2*sePt+1]);

	diff = patch1-patch2;
	diff2 = diff*diff;
	ssdValue = diff2.sum();
	#print ssdValue;
	return ssdValue;



def epipolarLine(points1, img2, points2,F,L2Rflag=False):
	#calculate and draw epipolar lines on the other image.
	result = img2.copy();
	height = img2.shape[1];
	width = img2.shape[0];
	for i in range(points1.shape[0]):
		interPts = []
		flags = []
		x = points1[i];
		line = np.dot(F,x);
		if L2Rflag:	
			line = np.dot(F.T,x);
		interPts.append([-line[2]/line[0], 0]);
		interPts.append([-(line[2]+line[1]*height)/line[0],height]);
		interPts.append([0,-line[2]/line[1]]);
		interPts.append([width,-(line[2]+line[0]*width)/line[1]]);
		startPt = [0,0];
		endPt = [0,0];
		for j in range(4):
			if j<2:
				if(interPts[j][0]>=0) and (interPts[j][0]<width):
					flags.append(1);
				else:
					flags.append(0);
			else:
				if(interPts[j][1]>=0) and (interPts[j][1]<height):
					flags.append(1);
				else:
					flags.append(0);
		ptFlag = 0;
		for j in range(4):
			if(flags[j]==1):
				if(ptFlag==0):
					startPt = np.int32(interPts[j]);
					ptFlag += 1;
				elif(ptFlag==1):
					endPt = np.int32(interPts[j]);
					ptFlag += 1;
				else:
					break;
		#print flags
		cv.line(result,(startPt[1],startPt[0]),(endPt[1],endPt[0]),(i*30,0,0),2);

	for i in range(points1.shape[0]):
		cv.circle(result, (points2[i][1],points2[i][0]), 5, (i*30,0,0));
	if(L2Rflag):
		cv.imwrite('wtfRight.jpg',result);
	else:
		cv.imwrite('wtfLineLeft.jpg',result);
	#print result.shape
	return result;



def calculEpi_n_CamMat(F):
	#calculate epipolars and camera matrices using Fundamental matrix
	U,D,V = np.linalg.svd(F);
	e = V[V.shape[0]-1];
	ep = U[:,U.shape[1]-1];
	epMat = vec2mat(ep);

	P = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]);
	Pp = np.dot(epMat,F);
	Pp = np.c_[Pp,ep];
	print 
	print "epipolars:";
	print e, '	', ep;
	print "P and P':"
	print P
	print Pp
	print
	return e,ep,P,Pp;



def vec2mat(vector):
	return np.array([[0,-vector[2],vector[1]],[vector[2],0,-vector[0]],[-vector[1],vector[0],0]]);


def Rectify(imgLeft,imgRight, F):
	#Image rectification. return homographies for rectification
	height = imgLeft.shape[1];
	width = imgLeft.shape[0];
	centerX = int(width/2);
	centerY = int(height/2);

	e,ep,P,Pp = calculEpi_n_CamMat(F);
	ep = ep/ep[2];
	e = e/e[2];

	T= np.array([[1,0,-centerX],[0,1,-centerY],[0,0,1]]);
	angle = math.atan(-(ep[1]-centerY)/(ep[0]-centerX));
	f = math.cos(angle)*(ep[0]-centerX)-math.sin(angle)*(ep[1]-centerY);
	R = np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]]);
	G= np.array([[1,0,0],[0,1,0],[-1/f,0,1]]);
	H2 = np.dot(np.dot(G,R),T);

	centerPt = np.array([centerX, centerY,1]);
	centerPt = np.dot(H2,centerPt);
	centerPt = centerPt/centerPt[2];
	T2 = np.array([[1,0,centerX-centerPt[0]],[0,1,centerY-centerPt[1]],[0,0,1]]);
	H2 = np.dot(T2,H2);
	H2 = H2/H2[2][2];

	ang = math.atan(-(e[1]-centerY)/(e[0]-centerX));
	f = math.cos(ang)*(e[0]-centerX)-math.sin(ang)*(e[1]-centerY);
	R = np.array([[math.cos(ang),-math.sin(ang),0],[math.sin(ang),math.cos(ang),0],[0,0,1]]);
	T = np.array([[1,0,-centerX],[0,1,-centerY],[0,0,1]]);
	G = np.array([[1,0,0],[0,1,0],[-1/f,0,1]]);
	Temp = np.dot(G,R)
	H1 = np.dot(np.dot(G,R),T);
	H1[2,0] = 0

	centerPt1 = np.array([centerX,centerY,1]);
	centerPt1 = np.dot(H1,centerPt1);
	centerPt1 = centerPt1/centerPt1[2];
	T1= np.array([[1,0,centerX-centerPt1[0]],[0,1,centerY-centerPt1[1]],[0,0,1]]);
	H1 = np.dot(T1,H1);

	return H1, H2;


def epipolarConstraints(matches,pointList1,pointList2, image1, image2):
	#using epipolar constraints (same row) and SSD to select correspondences in rectified images
	deleteList = [];
	for i in range(len(matches)):
		idxL = matches[i].queryIdx;
		idxR = matches[i].trainIdx;
		rowL = pointList1[idxL].pt[1];
		rowR = pointList2[idxR].pt[1];
		colL = pointList1[idxL].pt[0];
		colR = pointList2[idxR].pt[0];
		if abs(rowR-rowL)>17 or i==12 or i==57 or i==58 or i==59:
			deleteList.append(i);
		elif rowR<150 or rowL<150 or colL<100 or colR <100 or rowR>600  or rowL>600 or colL>650 or colR>650:
			deleteList.append(i);
		else:
			p1 = [int(colL),int(rowL)];
			p2 = [int(colR),int(rowR)];
			SSD = calSSD(p1,p2,image1,image2);
			if colL<200 and colL>180:
				deleteList.append(i);
			if SSD>1.2e6:
				deleteList.append(i);
	for remove in sorted(deleteList, reverse=True):
		del matches[remove];
	return matches;



def mapping2(image1,image2,H1,H2,interestPts1,interestPts2):
	#draw 2 images in the same canvas.
	#didn't use in the main code
	height = image1.shape[1];
	width = image1.shape[0];
	bd = np.array([[0,0,1],[width-1,0,1],[width-1,height-1,1],[0,height-1,1]]).T;
	H1 = np.matrix(H1);
	H2 = np.matrix(H2);

	bd1 = np.dot(H1,bd);
	bd1 = bd1/bd1[2];
	minX1 = bd1[0].min();
	minY1 = bd1[1].min();
	maxX1 = bd1[0].max();
	maxY1 = bd1[1].max();	
	width1 = int(maxX1-minX1);
	height1 = int(maxY1-minY1);

	bd2 = np.dot(H2,bd);
	bd2 = bd2/bd2[2];
	minX2 = bd2[0].min();
	minY2 = bd2[1].min();
	maxX2 = bd2[0].max();
	maxY2 = bd2[1].max();
	width2 = int(maxX2-minX2);
	height2 = int(maxY2-minY2);

	canvasMinY = minY1;
	canvasHeight = int(height1+height2);
	canvasMinX = min(minX1,minX2);
	canvasMaxX = max(maxX1,maxX2);
	canvasWidth = int(canvasMaxX - canvasMinX);
	
	offset1 = np.array([canvasMinX,minY1,0]);
	offset2 = np.array([canvasMinX,minY2,0]);
	result = np.zeros([canvasWidth,canvasHeight,3]);

	pts1 = np.dot(H1,interestPts1.T);
	pts1 = pts1/pts1[2];
	pts1 = pts1.T;
	pts2 = np.dot(H2,interestPts2.T);
	pts2 = pts2/pts2[2];
	pts2 = pts2.T;

	for i in range(0,canvasWidth-1):
		for j in range(0,height1-1):
			wld_coord = np.array([i,j,1])+offset1;
			img_coord = np.dot(np.array(H1.I),wld_coord);
			imgX = int(img_coord[0]/img_coord[2]);
			imgY = int(img_coord[1]/img_coord[2]);
			if(imgX>=0 and imgX<image1.shape[0] and imgY>=0 and imgY<image1.shape[1]):
				result[i,j] = image1[imgX,imgY];
			else:
				continue;

	for i in range(0,canvasWidth-1):
		for j in range(0,height2-1):
			wld_coord = np.array([i,j,1])+offset2;
			img_coord = np.dot(np.array(H2.I),wld_coord);
			imgX = int(img_coord[0]/img_coord[2]);
			imgY = int(img_coord[1]/img_coord[2]);
			if(imgX>=0 and imgX<image2.shape[0] and imgY>=0 and imgY<image2.shape[1]):
				result[i,j+height1] = image2[imgX,imgY];
			else:
				continue;
	for i in range(pts1.shape[0]):
		coord1 = pts1[i]-offset1;
		coord2 = pts2[i]-offset2+np.array([0,height1,0]);
		print coord1[0,0],coord1[0,1]
		cv.circle(result, (int(coord1[0,0]),int(coord1[0,1])), 5, (i*30,0,255));
		cv.circle(result, (int(coord2[0,0]),int(coord2[0,1])), 5, (0,255,0));
	return result;



def mapping(image,homography,points):
	#map the image to a proper canvas using given homography
	dstCanvas = np.array([[0,0,1],[0,image.shape[1],1],[image.shape[0],image.shape[1],1],[image.shape[0],0,1]]);
	wldCanvas = np.dot(homography,dstCanvas.T);
	wldCanvas[0] = wldCanvas[0]/wldCanvas[2];
	wldCanvas[1] = wldCanvas[1]/wldCanvas[2];
	wldCanvas[2] = wldCanvas[2]/wldCanvas[2];
	
	maxH = wldCanvas[0].max();
	minH = wldCanvas[0].min();
	maxW = wldCanvas[1].max();
	minW = wldCanvas[1].min();
	
	offset = [int(minH),int(minW)];
	canvas = [int(math.ceil(maxH-minH)),int(math.ceil(maxW-minW))];	
	
	result = np.zeros([canvas[0],canvas[1],3]);
	
	for i in range(0,canvas[0]-1):
		for j in range(0,canvas[1]-1):
			wld_coord = np.array([i+offset[0],j+offset[1],1]);
			img_coord = np.dot(np.array(np.matrix(homography).I),wld_coord);
			imgX = int(img_coord[0]/img_coord[2]);
			imgY = int(img_coord[1]/img_coord[2]);
			if(imgX>=0 and imgX<image.shape[0] and imgY>=0 and imgY<image.shape[1]):
				result[i,j] = image[imgX,imgY];
			else:
				continue;
	coords = [];
	for i in range(points.shape[0]):
		wld_coord = np.dot(homography,points[i]);
		wld_coord = wld_coord/wld_coord[2];
		coord = [int(wld_coord[0])-offset[0],int(wld_coord[1])-offset[1],1];
		#cv.circle(result, (coord[1],coord[0]), 5, (0,0,255));
		coords.append(coord);
	coords = np.array(coords);
	return result,coords;



def normalizePts(points):
	#normalize the selected points so their mean will be the origin and distance is sqrt(2)
	#the results were worse with nomalization. didn't use at last
	meanPt = points.mean(axis=0);
	diffS = points - meanPt;
	varS = diffS**2;
	distS = (varS[:,0]+varS[:,1])**0.5;
	dist = distS.mean();
	
	scale = 2**0.5/dist;
	trans = -scale*meanPt;
	T = np.array([[scale,0,trans[0]],[0,scale,trans[1]],[0,0,1]]);

	nmlzPts = [];
	for i in range(points.shape[0]):
		nmlzPt = np.dot(T,points[i]);
		nmlzPt = nmlzPt/nmlzPt[2];
		nmlzPts.append(nmlzPt);
	nmlzPts = np.array(nmlzPts);
	return nmlzPts, T;



def findF(manuallyPts_left, manuallyPts_right):
	#coords_left, T1 = normalizePts(manuallyPts_left);
	#coords_right, T2 = normalizePts(manuallyPts_right);
	coords_left = manuallyPts_left;
	coords_right = manuallyPts_right;

	A = np.zeros((coords_left.shape[0],9));
	F = np.zeros((3,3));
		
	for i in range(coords_left.shape[0]):
		A[i] = [coords_right[i][0]*coords_left[i][0],coords_right[i][0]*coords_left[i][1],coords_right[i][0],coords_right[i][1]*coords_left[i][0],coords_right[i][1]*coords_left[i][1],coords_right[i][1], coords_left[i][0], coords_left[i][1], 1];
	
	U,D,V = np.linalg.svd(A);
	f = V[V.shape[0]-1];
	for i in range(f.size):
		F[i/3,i%3] = f[i];
	U,D,V = np.linalg.svd(F);
	D[D.size-1]=0;
	F = np.dot(np.dot(U,np.identity(D.size)*D),V);
	#F = np.dot(np.dot(T2.T,F),T1);
	F = F/F[2][2];
	print "Estimated F: "
	print F
	return F, coords_left, coords_right#, T1, T2;



def findHomographyHQ(coords_range, coords_domain):
	#the same function used before
	#used for testing the homogeneos solve only
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
	#for ransac
	#didn't use
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
	return H, inliers1, inliers2, T1, T2;



def ransacHQ(interestPoints1,interestPoints2, eps, p, n, delta):
	#didn't use
	#the same function in hw5 without modified
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
	#same function in hw5
	#didn't use
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


