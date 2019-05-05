import numpy as np

from lr_utils import load_dataset


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

def layer_sizes(X,Y):
	n_x=X.shape[0]
	n_h=4
	n_y=Y.shape[0]
	return (n_x,n_h,n_y)

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def initialize_parameters(n_x,n_h,n_y):

	np.random.seed(2)

	W1=np.random.randn(n_h,n_x)*0.01
	b1=np.zeros((n_h,1))
	W2=np.random.randn(n_y,n_h)*0.01
	b2=np.zeros((n_y,1))

	parameters={"W1":W1,
				"b1":b1,
				"W2":W2,
				"b2":b2}

	return parameters


def forward_propagation(X,parameters):

	W1=parameters["W1"]
	b1=parameters["b1"]
	W2=parameters["W2"]
	b2=parameters["b2"]

	Z1=np.dot(W1,X)+b1
	A1=np.tanh(Z1)
	Z2=np.dot(W2,A1)+b2
	A2=sigmoid(Z2)

	cache={"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}

	return A2,cache

def compute_cost(A2,Y,parameters):
	m=Y.shape[1]

	logprobs=np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),(1-Y))
	cost=-np.sum(logprobs)/m

	cost=np.squeeze(cost)

	return cost


def backward_propagation(parameters,cache,X,Y):
	m=X.shape[1]

	W1=parameters["W1"]
	W2=parameters["W2"]

	A1=cache["A1"]
	A2=cache["A2"]

	dZ2=A2-Y
	dW2=np.dot(dZ2,A1.T)/m
	db2=np.sum(dZ2,axis=1,keepdims=True)/m
	dZ1=np.multiply(np.dot(W2.T,dZ2),(1-np.power(A1,2)))
	dW1=np.dot(dZ1,X.T)/m
	db1=np.sum(dZ1,axis=1,keepdims=True)/m

	grads={"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2}

	return grads


def update_parameters(parameters,grads,learning_rate=1.2):
	W1=parameters["W1"]
	W2=parameters["W2"]
	b1=parameters["b1"]
	b2=parameters["b2"]

	dW1=grads["dW1"]
	db1=grads["db1"]
	dW2=grads["dW2"]
	db2=grads["db2"]

	W1=W1-learning_rate*dW1
	b1=b1-learning_rate*db1
	W2=W2-learning_rate*dW2
	b2=b2-learning_rate*db2

	parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2}

	return parameters

def snn_model(X,Y,n_h,num_iterations=2000,print_cost=False):

	np.random.seed(3)
	n_x=layer_sizes(X,Y)[0]
	n_y=layer_sizes(X,Y)[2]

	parameters=initialize_parameters(n_x,n_h,n_y)
	W1=parameters["W1"]
	W2=parameters["W2"]
	b1=parameters["b1"]
	b2=parameters["b2"]

	for i in range(0, num_iterations):
		A2,cache=forward_propagation(X,parameters)
		cost=compute_cost(A2,Y,parameters)
		grads=backward_propagation(parameters,cache,X,Y)
		parameters=update_parameters(parameters,grads,learning_rate=0.0075)
		if print_cost and i % 100 == 0:
			print ("Cost after iteration %i: %f" %(i, cost))

	return parameters

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2>0.5)
    ### END CODE HERE ###
    
    return predictions


parameters=snn_model(train_set_x,train_set_y,8,print_cost=True)

Y_prediction_train=predict(parameters,train_set_x)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
Y_prediction_test=predict(parameters,test_set_x)
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))

