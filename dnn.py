import numpy as np 
import h5py
from dnn_utils_v2 import *

def load_data():
    train_dataset = h5py.File('/Users/zhangshihao/desktop//DL_Week4/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('/Users/zhangshihao/desktop//DL_Week4/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def initialize_parameters(layer_dims):
	np.random.seed(1)

	parameters = {}
	L=len(layer_dims)

	for l in range(1,L):
		parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
		parameters["b"+str(l)]=np.zeros((layer_dims[l],1))

	return parameters


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
	 	A,cache=linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation="relu")
	 	caches.append(cache)

	 AL, cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation="sigmoid")
	 caches.append(cache)

	 return AL,caches

def compute_cost(AL,Y):
	m=Y.shape[1]

	cost = -np.sum(np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL)))/m

	cost = np.squeeze(cost)

	return cost

def linear_backward(dZ,cache):

	A_prev,W,b=cache
	m=A_prev.shape[1]

	dW = np.dot(dZ,A_prev.T)/m
	db = np.sum(dZ,axis=1,keepdims=True)/m
	dA_prev = np.dot(W.T,dZ)

	return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):

	linear_cache,activation_cache=cache

	if activation == "relu":
		dZ = relu_backward(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,linear_cache)
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,linear_cache)

	return dA_prev,dW,db

def L_model_backward(AL,Y,caches):

	grads = {}
	L=len(caches)
	m=AL.shape[1]
	Y=Y.reshape(AL.shape)

	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	current_cache = caches[L-1]
	grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_backward(dAL,current_cache,activation="sigmoid")

	for l in reversed(range(L-1)):
		current_cache=caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,activation="relu")
		grads["dA"+str(l)]=dA_prev_temp
		grads["dW"+str(l+1)]=dW_temp
		grads["db"+str(l+1)]=db_temp

	return grads

def update_parameters(parameters,grads,learning_rate):

	L=len(parameters)//2

	for l in range(L):
		parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
		parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-grads["db"+str(l+1)]*learning_rate

	return parameters

def L_layer_model(X,Y,layer_dims,learning_rate=0.0075,num_iterations=3000,print_cost=True):
	np.random.seed(1)
	cost=[]

	parameters=initialize_parameters(layer_dims)

	for i in range(0,num_iterations):

		AL,caches = L_model_forward(X,parameters)

		cost=compute_cost(AL,Y)

		grads = L_model_backward(AL, Y, caches)

		parameters = update_parameters(parameters, grads, learning_rate)

		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))

	return parameters

def predict(X, Y, parameters,datasets="train"):

    # Forward propagation
    AL, caches = L_model_forward(X, parameters)
    predictions= (AL>0.5)
    
    if datasets == "train":
        print("Train Accuracy: {}% ".format(100-np.mean(np.abs(predictions-Y))*100))
    elif datasets == "test":
        print("Test Accuracy: {}%".format(100-np.mean(np.abs(predictions-Y))*100))

    return predictions

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

layer_dims = [12288, 10, 5, 1]
parameters = L_layer_model(train_x, train_y, layer_dims,learning_rate=0.01,num_iterations =2000, print_cost = True)
pred_train = predict(train_x, train_y, parameters,"train")
pred_test = predict(test_x, test_y, parameters,"test")
