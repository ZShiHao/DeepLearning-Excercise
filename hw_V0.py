import numpy as np 
import matplotlib.pyplot as plt
import time
from mnist_loader import *
from dnn_utils_v2 import *

def initialize_parameters(layer_dims):
	np.random.seed(1)

	parameters = {}
	L=len(layer_dims)

	for l in range(1,L):
		parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(1/layer_dims[l-1])
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

	cost_AL= np.sum(np.multiply(Y,np.log(AL))+np.multiply((1-Y),np.log(1-AL)),axis=0)
	cost=-np.sum(cost_AL)/m

	cost = np.squeeze(cost)

	return cost

def compute_cost_regularzition(cross_entropy_cost,parameters,lambd=0.7):
	m=len(parameters)//2
	L2_cost=0

	for l in range(0,m):
		L2_cost=L2_cost+np.squeeze(np.sum(np.square(parameters["W"+str(l+1)])))

	L2_cost=lambd*L2_cost/(2*m)

	L2_cost=L2_cost+cross_entropy_cost

	return L2_cost


def linear_backward(dZ,lambd,cache):

	A_prev,W,b=cache
	m=A_prev.shape[1]

	dW = np.dot(dZ,A_prev.T)/m+lambd*W/m
	db = np.sum(dZ,axis=1,keepdims=True)/m
	dA_prev = np.dot(W.T,dZ)

	return dA_prev,dW,db

def linear_activation_backward(dA,cache,lambd,activation):

	linear_cache,activation_cache=cache

	if activation == "relu":
		dZ = relu_backward(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,lambd,linear_cache)
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,lambd,linear_cache)

	return dA_prev,dW,db

def L_model_backward(AL,Y,lambd,caches):

	grads = {}
	L=len(caches)
	m=AL.shape[1]
	Y=Y.reshape(AL.shape)

	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

	current_cache = caches[L-1]
	grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_backward(dAL,current_cache,lambd,activation="sigmoid")

	for l in reversed(range(L-1)):
		current_cache=caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,lambd,activation="relu")
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

def L_layer_model(X,Y,layer_dims,learning_rate=0.0075,num_iterations=3000,lambd=0.7,print_cost=True):
	np.random.seed(1)
	costs=[]
	iterations=[]

	parameters=initialize_parameters(layer_dims)

	for i in range(0,num_iterations):

		AL,caches = L_model_forward(X,parameters)

		cross_entropy_cost=compute_cost(AL,Y)

		grads = L_model_backward(AL, Y, lambd,caches)

		parameters = update_parameters(parameters, grads, learning_rate)

		if print_cost and i % 10 == 0:
			print ("Cost after iteration %i: %f" %(i, cross_entropy_cost))
		if print_cost and i % 10 == 0:
			iteration=i
			iterations.append(iteration)
			costs.append(cross_entropy_cost)

	plt.plot(iterations,costs)
	plt.ylabel('cost')
	plt.xlabel('iterations ')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

	return parameters

def predict(X, Y, parameters,datasets="train"):

    # Forward propagation
    AL, caches = L_model_forward(X, parameters)
    y=np.argmax(Y,axis=0)
    predictions= np.argmax(AL,axis=0)
    
    if datasets == "train":
        print("Train Accuracy: {}% ".format((np.sum([ int(i) for i in y==predictions])/len(y))*100))
    elif datasets == "test":
        print("Test Accuracy: {}%".format((np.sum([ int(i) for i in y==predictions])/len(y))*100))
    return predictions

training_data,validation_data,test_data=load_data()
training_inputs,training_results=training_data
train_x=training_inputs.T
train_y=vectorized_result(training_results)

layer_dims = [784, 30,20,10]
start=time.time()
parameters = L_layer_model(train_x, train_y, layer_dims,learning_rate=3,num_iterations =500, lambd=0,print_cost = True)
end=time.time()
print("Training time : %f s"%(end-start))
pred_train = predict(train_x, train_y, parameters,"train")

validation_inputs,validation_results=validation_data
test_x=validation_inputs.T
test_y=vectorized_result(validation_results)
pred_test = predict(test_x, test_y, parameters,"test")


