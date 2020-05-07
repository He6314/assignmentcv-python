#Some functions for basic homography and other transform matrices
#by Qin He

import numpy as np
import math

def findHomography(coords_range, coords_domain):
	#calculate the point-to-point homography using 4 given point pairs
	paraMat = np.zeros((8,8));
	paraMat = np.matrix(paraMat)
	b = np.zeros((1,8));
	H = np.zeros((3,3));
		
	for i in range(0,len(coords_range)):
		paraMat[i*2] = [coords_range[i][0],coords_range[i][1],coords_range[i][2],0,0,0,(-1*coords_range[i][0]*coords_domain[i][0]),(-1*coords_range[i][1]*coords_domain[i][0])];
		paraMat[i*2+1] = [0,0,0,coords_range[i][0],coords_range[i][1],coords_range[i][2],(-1*coords_range[i][0]*coords_domain[i][1]),(-1*coords_range[i][1]*coords_domain[i][1])];
		b[0][i*2] = coords_domain[i][0];
		b[0][i*2+1] = coords_domain[i][1];	
	
	vecH = np.dot(paraMat.I,b.T);	
	for i in range(0,len(vecH)):
		H[i/3,i%3] = vecH[i];
	H[2,2] = 1;

	return H;


def findAffine(lineH1,lineV1,lineH2,lineV2):
	#calculate homography that resulting affine distortions
	coefficient = np.matrix([[lineH1[0]*lineV1[0],lineH1[0]*lineV1[1]+lineH1[1]*lineV1[0]],[lineH2[0]*lineV2[0],lineH2[0]*lineV2[1]+lineH2[1]*lineV2[0]]]);
	product = np.array([-1*lineH1[1]*lineV1[1],-1*lineH2[1]*lineV2[1]]);
	
	sVector = np.dot(coefficient.I,product.T);
	s = np.matrix([[sVector[0,0],sVector[0,1]],[sVector[0,1],1]]);	
	V,Dsq,U = np.linalg.svd(s);
	D = np.array([[math.sqrt(Dsq[0]),0],[0,math.sqrt(Dsq[1])]]);
	
	a = np.dot(np.dot(V,D),V.T);
	A = np.array([[a[0,0],a[0,1],0],[a[1,0],a[1,1],0],[0,0,1]]);
	
	return A;


def orthPairCfft(line1,line2):
	#use orthorogy line pair to construct the coeffecients for dual degenerate conic matrix
	c1 = line1[0]*line2[0];
	c2 = (line1[0]*line2[1] + line1[1]*line2[0])/2;
	c3 = line1[1]*line2[1];
	c4 = (line1[0]*line2[2] + line1[2]*line2[0])/2;
	c5 = (line1[1]*line2[2] + line1[2]*line2[1])/2;
	c6 = -line1[2]*line2[2];#the products (-1)s

	return np.array([c1,c2,c3,c4,c5,c6]);


def findDDConicHmg(coefficient_n_product):
	#use given coefficients to calculate the (distorted) dual degenarate conic, and then the homography H in {img=H*wld}
	product = coefficient_n_product[:,5];
	coefficient = coefficient_n_product[:,0:5];

	hVector = np.dot(coefficient.I,product);
	hVector /= max(hVector);
	hVector = hVector.T;
	
	S = np.array([[hVector[0,0],hVector[0,1]/2,hVector[0,3]/2],[hVector[0,1]/2,hVector[0,2],hVector[0,4]/2],[hVector[0,3]/2,hVector[0,4]/2,1]]);
	
	s1 = S[0:2,0:2];
	V,Dsq,V = np.linalg.svd(s1);
	D = np.matrix([[math.sqrt(Dsq[0]),0],[0,math.sqrt(Dsq[1])]]);
	a = np.dot(np.dot(V,D),V.T);

	s2 = S[2,0:2];
	v = np.dot(a.I,s2);
	
	H = np.array([[a[0,0],a[0,1],0],[a[1,0],a[1,1],0],[v[0,0],v[0,1],1]]);
	return H


def mapping(image,homography):
	#map the image to a proper canvas using given homography
	dstCanvas = np.array([[0,0,1],[0,image.shape[1],1],[image.shape[0],image.shape[1],1],[image.shape[0],0,1]]);
	wldCanvas = np.dot(np.matrix(homography).I,dstCanvas.T);
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
			img_coord = np.dot(homography,wld_coord);
			imgX = int(img_coord[0]/img_coord[2]);
			imgY = int(img_coord[1]/img_coord[2]);
			if(imgX>=0 and imgX<image.shape[0] and imgY>=0 and imgY<image.shape[1]):
				result[i,j] = image[imgX,imgY];
			else:
				continue;
	
	return result;

