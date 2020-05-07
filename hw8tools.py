#Functions for camera calibration
#by Qin He

import numpy as np
import cv2 as cv
import math

def trimLines(lineKBhori,lineKBverti,lineHori,lineVerti,pruneThresh):
	for i in range(len(lineKBhori)-1,-1,-1):
		line1 = lineKBhori[i];
		for j in range(i):
			line2 = lineKBhori[j];
			distB = abs(line1[1]-line2[1]);
			if(distB<pruneThresh):
				del lineKBhori[i];
				del lineHori[i];
				break;
	print 'HORIZON:',len(lineKBhori)

	for i in range(len(lineKBverti)-1,-1,-1):
		line1 = lineKBverti[i];
		for j in range(i):
			line2 = lineKBverti[j];
			distB = abs(line1[1]-line2[1]);
			if(distB<pruneThresh):
				del lineKBverti[i];
				del lineVerti[i];
				break;
	print 'VERTICAL:',len(lineKBverti)

	lineHori.sort(key=lambda x:x[4]);
	lineVerti.sort(key=lambda x:x[4]);


def HoughCorners(imageRGB,number=0,cannyThresh1=150, cannyThresh2=250, gz=5, dataset='1',write=False):
	image = cv.cvtColor(imageRGB, cv.COLOR_BGR2GRAY);
	image = cv.GaussianBlur(image,(gz,gz),0);
	imageCopy = imageRGB.copy();

#	houghThresh = image.shape[0]/20;
#	minLineLength = image.shape[0]/5;
#	maxLineGap = image.shape[0]/6;
#	pruneThresh = houghThresh/2;

	if(dataset=='1'):
		houghThresh = 30;
		minLineLength = 90;
		maxLineGap = 70;
		pruneThresh = houghThresh/2;

	elif(dataset=='2'):
		houghThresh = 25;
		minLineLength = 75;
		maxLineGap = 65;
		pruneThresh = houghThresh/1.8;

	edge = cv.Canny(image, cannyThresh1, cannyThresh2);
	if(write):
		cv.imwrite(dataset+'/edges/'+str(number)+'.jpg',edge);
	
	lines = cv.HoughLinesP(edge,1.0,0.7*np.pi/180.0,houghThresh,0,minLineLength,maxLineGap);
	lines = cv.HoughLinesP(edge,1.0,0.7*np.pi/180.0,houghThresh,0,minLineLength,maxLineGap);
	N = lines.shape[0]
	print 'BEFORE_N:',N

	lineKBhori = [];
	lineHori = [];
	lineKBverti = [];
	lineVerti = [];
	for i in range(N):
		point1 = np.array([lines[i][0][0],lines[i][0][1],1],float);
		point2 = np.array([lines[i][0][2],lines[i][0][3],1],float);
		if ((point2[0]-point1[0])<1):
			slope = image.shape[0];
		else:
			slope = (point2[1]-point1[1])/(point2[0]-point1[0]);
		intercept = point1[1]-point1[0]*slope;
		if(abs(slope)<1):
			lineKBhori.append([slope,intercept,i]);
			lineHori.append([lines[i][0][0],lines[i][0][1],lines[i][0][2],lines[i][0][3],intercept]);
		else:
			intercept = (-intercept)/slope;
			lineKBverti.append([slope,intercept,i]);
			lineVerti.append([lines[i][0][0],lines[i][0][1],lines[i][0][2],lines[i][0][3],intercept]);
	print 'BEFORE_H,V:',len(lineKBhori),len(lineKBverti)


	trimLines(lineKBhori,lineKBverti,lineHori,lineVerti,pruneThresh);
	print 'AFTER_N:', len(lineHori)+len(lineVerti)

	for i in range(len(lineHori)):
		x1 = lineHori[i][0];
		y1 = lineHori[i][1];   
		x2 = lineHori[i][2];
		y2 = lineHori[i][3];
		color = (255,0,0);
		cv.line(imageCopy,(x1,y1),(x2,y2),color,2);
	for i in range(len(lineVerti)):
		x1 = lineVerti[i][0];
		y1 = lineVerti[i][1];   
		x2 = lineVerti[i][2];
		y2 = lineVerti[i][3];
		color = (0,255,0);
		cv.line(imageCopy,(x1,y1),(x2,y2),color,2);
	if(write):
		cv.imwrite(dataset+'/lines/'+str(number)+'.jpg',imageCopy);
	if(write):
		if(len(lineHori)!=10)or(len(lineVerti)!=8):
			cv.imwrite(dataset+'/errors/'+str(number)+'.jpg',imageCopy);

	edge = cv.cvtColor(edge,cv.COLOR_GRAY2BGR);
	corners = [];
	for i in range(len(lineHori)):
		for j in range(len(lineVerti)):
			point11 = np.array([lineHori[i][0],lineHori[i][1],1],float);
			point12 = np.array([lineHori[i][2],lineHori[i][3],1],float);
			line1 = np.cross(point11,point12);
			line1 /= line1[2];
			point21 = np.array([lineVerti[j][0],lineVerti[j][1],1],float);
			point22 = np.array([lineVerti[j][2],lineVerti[j][3],1],float);
			line2 = np.cross(point21,point22);
			line2 /= line2[2];
			interPt = np.cross(line1,line2);
			if(abs(interPt[2])>0):
				interPt /= interPt[2];
				interPt = np.int32(interPt);
				corners.append(interPt);
			cv.circle(edge, (interPt[0],interPt[1]), 5, (255,int(float(j)/8*255),int(float(i)/10*255)));
			cv.putText(edge,str(i*8+j+1),(interPt[0]+2,interPt[1]+2),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

	if(write):
		cv.imwrite(dataset+'/corners/'+str(number)+'.jpg',edge);
	
	print 'NUM_CORNERS:', len(corners)
	print 'CORNER DETECTION END======================'
	return corners, lineHori, lineVerti;


def findHomographyHQ(coords_domain, coords_range):
	#the same function used in Homework5
	print 'image len:',len(coords_domain)
	print 'world len:',len(coords_range)
	A = np.zeros((2*len(coords_domain),9));
	H = np.zeros((3,3));
		
	for i in range(len(coords_domain)):
		A[i*2] = [0,0,0,-coords_domain[i][0],-coords_domain[i][1],-1, coords_domain[i][0]*coords_range[i][1], coords_domain[i][1]*coords_range[i][1], coords_range[i][1]];
		A[i*2+1] = [coords_domain[i][0],coords_domain[i][1],1,0,0,0,(-1*coords_domain[i][0]*coords_range[i][0]),(-1*coords_domain[i][1]*coords_range[i][0]), -coords_range[i][0]];
	
	U,Dsq,V = np.linalg.svd(A);
	vecH = V[V.shape[0]-1];
	for i in range(vecH.size):
		H[i/3,i%3] = vecH[i];
	return H;


def findIntrinPara(hList):
	N = len(hList);
	zhangV = [];
	for n in range(N):
		H = hList[n];
		h = [];
		h.append(H[:,0]);
		h.append(H[:,1]);
		for i in range(2):
			for j in range(2):
				vSub = [h[i][0]*h[j][0],h[i][0]*h[j][1]+h[i][1]*h[j][0],h[i][1]*h[j][1],h[i][2]*h[j][0]+h[i][0]*h[j][2],h[i][2]*h[j][1]+h[i][1]*h[j][2],h[i][2]*h[j][2]]
				if(i==0 and j==1):
					v1 = np.array(vSub);
				elif(i==0 and j==0):
					v2 = np.array(vSub);
				elif(i==1 and j==1):
					v2 -= np.array(vSub);
		zhangV.append(v1);
		zhangV.append(v2);

	zhangV = np.array(zhangV);
	print 'SHAPE Vmat:', zhangV.shape

	omega = np.zeros((3,3));
	
	U,Dsq,V = np.linalg.svd(zhangV);
	vecOmega = V[V.shape[0]-1];

	omega[0,0] = vecOmega[0];
	omega[0,1] = vecOmega[1];
	omega[0,2] = vecOmega[3];
	omega[1,0] = vecOmega[1];
	omega[1,1] = vecOmega[2];
	omega[1,2] = vecOmega[4];
	omega[2,0] = vecOmega[3];
	omega[2,1] = vecOmega[4];
	omega[2,2] = vecOmega[5];

	x0 = (omega[0,1]*omega[0,2]-omega[0,0]*omega[1,2])/(omega[0,0]*omega[1,1]-omega[0,1]*omega[0,1]);
	lam = omega[2,2]-(omega[0,2]*omega[0,2]+x0*(omega[0,1]*omega[0,2]-omega[0,0]*omega[1,2]))/omega[0,0];
	alphaX = (lam/omega[0,0])**0.5;
	alphaY = (lam*omega[0,0]/(omega[0,0]*omega[1,1]-omega[0,1]*omega[0,1]))**0.5;
	s = -(omega[0,1]*alphaX*alphaX*alphaY/lam);
	y0 = s*x0/alphaY - omega[0,2]*alphaX*alphaX/lam;

	K = np.matrix([[alphaX,s,x0],[0,alphaY,y0],[0,0,1]]);
	intrinPara = [x0,y0,alphaX,alphaY,s];
	print 'FOUND K:' 
	print K
	return K,intrinPara;


def findExtrinPara(H,Kinv):
	h1 = np.matrix(H[:,0]).T;
	h2 = np.matrix(H[:,1]).T;
	h3 = np.matrix(H[:,2]).T;
	
	t = np.dot(Kinv,h3);
	xi = 1/np.linalg.norm(Kinv*h1);
	if(t[2]<0):
		xi = -xi;  #the camera can't locate behind the patter;

	r1 = xi*np.dot(Kinv,h1);
	r2 = xi*np.dot(Kinv,h2);
	r3 = np.cross(r1.T,r2.T).T;
	t = xi*t;

	R = np.hstack((r1,r2,r3,t))

	print '';
	print 'FOUND [R|t]: ';
	#print [np.linalg.norm(r1),np.linalg.norm(r2),np.linalg.norm(r3)]
	#print np.hstack((r1,r2,r3)).trace();
	print R
	return R;


def Rodriguez(R):
	if (R.shape[0]==1):
		wMat = np.matrix([[0,-R[0,2],R[0,1]],[R[0,2],0,-R[0,0]],[-R[0,1],R[0,0],0]]);
		phi = np.linalg.norm(R);
		rodMatrix = np.identity(3)+math.sin(phi)/phi*wMat+(1-math.cos(phi))/(phi*phi)*(wMat**2);
		return rodMatrix;
	else:
		traceR = R.trace();
		#if(traceR<-1):
			#traceR = -1; #some of the traces may be smaller than -1, I think it may relate to the precision of floating number.
		phi = math.acos((traceR-1)/2);
		wVec = np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[0,1]-R[1,0]]);
		rodVector = phi/(2*math.sin(phi))*wVec;
		return rodVector;

























def prunePts(corners,shape,threshOverlap,threshIsolate):
	for i in range(len(corners)-1,-1,-1):	
		if corners[i][0]<0 or corners[i][1]<0 or corners[i][0]>shape[1] or corners[i][1]>shape[0]: 
			del corners[i];

	for i in range(len(corners)-1,-1,-1):	
		pt1 = corners[i];
		for j in range(i):
			pt2 = corners[j];
			diff = pt1-pt2;
			dist = (diff[0]**2+diff[1]**2)**0.5;
			if(dist<threshOverlap):
				del corners[i];
				delFlag = True;
				break;

	for n in range(2):
		for i in range(len(corners)-1,-1,-1):
			pt1 = corners[i];
			nei = 0;
			for j in range(len(corners)):
				pt2 = corners[j];
				dist = np.linalg.norm(pt1 - pt2);
				if(dist<threshIsolate):
					nei += 1;
			if(nei<3):
				del corners[i];


def HoughCornersPrunePts(imageRGB,threshold1,threshold2,gz):
	image = cv.cvtColor(imageRGB, cv.COLOR_BGR2GRAY);

	Threshold = image.shape[0]/20;
	minLineLength = image.shape[0]/5;
	maxLineGap = image.shape[0]/6;

	image = cv.GaussianBlur(image,(gz,gz),0);
	edge = cv.Canny(image, threshold1, threshold2);
	lines = cv.HoughLinesP(edge,1.0,1.0*np.pi/180.0,Threshold,0,minLineLength,maxLineGap);
	N = lines.shape[0]
	
	corners = [];
	for i in range(N):
		for j in range(i+1,N):
			point11 = np.array([lines[i][0][0],lines[i][0][1],1],float);
			point12 = np.array([lines[i][0][2],lines[i][0][3],1],float);
			line1 = np.cross(point11,point12);
			line1 /= line1[2];
			point21 = np.array([lines[j][0][0],lines[j][0][1],1],float);
			point22 = np.array([lines[j][0][2],lines[j][0][3],1],float);
			line2 = np.cross(point21,point22);
			line2 /= line2[2];
			interPt = np.cross(line1,line2);
			if(abs(interPt[2])>0):
				interPt /= interPt[2];
				interPt = np.int32(interPt);
				corners.append(interPt);

	prune(corners,image.shape,Threshold/1.5,Threshold*2.2);
	print len(corners);

	for i in range(len(corners)):
		cv.circle(imageRGB, (corners[i][0],corners[i][1]), 5, (0,0,255));
		#cv.circle(imageRGB, (corners[i][0],corners[i][1]), Threshold*3, (255,0,0));	




