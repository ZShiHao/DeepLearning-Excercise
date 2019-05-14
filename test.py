import pickle
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import numpy as np
from dnn_utils_v2 import *

def linear_forward(A_prev,W,b):

	#Z=np.dot(W,A_prev)+b
	Z = W.dot(A_prev) + b

	cache=(A_prev,W,b)

	return Z,cache

def linear_activation_forward(A_prev,W,b,activation):

	if activation == "sigmoid":
		Z,linear_cache = linear_forward(A_prev,W,b)
		A,activation_cache = sigmoid(Z)
	elif activation == "relu":
		Z,linear_cache = linear_forward(A_prev,W,b)
		A,activation_cache = relu(Z)

	cache = (linear_cache,activation_cache)

	return A,cache

def L_model_forward(X,parameters):
	 caches=[]
	 A=X
	 L=len(parameters)//2

	 for l in range(1,L):
	 	A_prev = A 
	 	A,cache=linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation="sigmoid")
	 	caches.append(cache)

	 AL, cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation="sigmoid")
	 caches.append(cache)

	 return AL,caches

def ROI(image_array):
	size_row=[]
	size_column=[]
	I=image_array
	for i in range(2,I.shape[0]-2):
	    k=I[i,2:I.shape[1]-2]<150
	    if k.all() :
	        pass
	    else :
	        size_row.append(i-5)
	        break
	for j in range(I.shape[0]-3,3,-1):
	    k=I[j,2:I.shape[1]-2]<150
	    if k.all() :
	        pass
	    else :
	        size_row.append(j+5)
	        break

	for i in range(2,I.shape[1]-2):
	    k=I[2:I.shape[0]-2,i]<150
	    if k.all() :
	        pass
	    else :
	        size_column.append(i-5)
	        break
	for j in range(I.shape[1]-3,3,-1):
	    n=I[2:I.shape[0]-2,j]<150
	    if n.all() :
	        pass
	    else :
	        size_column.append(j+5)
	        break
	return I,size_row,size_column

def Square(image_array,size_row,size_column):
	I=image_array
	x=size_row[1]-size_row[0]
	y=(size_column[1]-size_column[0])
	d=x-y
	if d>1:
	    if d%2==0:
	        size_column[0]=int(size_column[0]-d/2)
	        size_column[1]=int(size_column[1]+d/2)
	    else :
	        size_column[0]=int(size_column[0]-(d+1)/2)
	        size_column[1]=int(size_column[1]+(d+1)/2)
	elif d<-1:
	    if (-d)%2==0:
	        size_row[0]=int(size_row[0]-(-d)/2)
	        size_row[1]=int(size_row[1]+(-d)/2)
	    else :
	        size_row[0]=int(size_row[0]-(-d+1)/2)
	        size_row[1]=int(size_row[1]+(-d+1)/2)

	I=I[size_row[0]:size_row[1],size_column[0]:size_column[1]]
	I=Image.fromarray(I)
	I=I.resize((28,28))
	scipy.misc.imsave('./processed2.png', I)

	return I


def Image_process(image_name):
	I=Image.open(image_name)
	I=I.convert("L")
	I=ImageOps.invert(I)
	I=np.array(I)
	I=I
	a=I<100
	I[a]=0.0

	I,size_row,size_column=ROI(I)
	image=Square(I,size_row,size_column)
	return image



f=open('./model_V1.pickle','rb')
parameters = pickle.load(f,encoding='latin1')
f.close()


image_name='./9.jpeg'
image=Image_process(image_name)
I=np.reshape(np.array(image)/255,(784,1))

AL,caches=L_model_forward(I,parameters)

x=np.argmax(AL)

print("The number on this picture is {}.".format(x))
plt.imshow(image)
plt.title("The number on this picture is {}.".format(x))
plt.show()


