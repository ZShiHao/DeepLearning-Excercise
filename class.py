import pickle
import math
import scipy.misc
from PIL import Image,ImageOps
from functools import reduce
import numpy as np
import time
from dnn_utils_v2 import *
from mnist_loader import *

class Conv_layer(object):
    def __init__(self, shape, output_channels, ksize=3, stride=1, method='SAME'):
        self.shape=shape
        self.input_shape = self.shape
        self.output_channels = output_channels
        self.input_channels = self.shape[-1]
        self.batchsize = self.shape[0]
        self.stride = stride
        self.ksize = ksize
        self.method = method

        weights_scale = math.sqrt(reduce(lambda x, y: x * y, self.shape) / self.output_channels)
        for c in range(0,self.output_channels):
            np.random.seed(c)
            for C_prev in range(0,self.input_channels):
                filter2D=np.random.randn(ksize,ksize)
                filter3D=filter2D[:,:,np.newaxis].repeat(self.input_channels,axis=2)
                filter4D=filter3D[:,:,:,np.newaxis]
            if c==0:
                self.weights=filter4D
            else:
                self.weights=np.concatenate((self.weights,filter4D),axis=3)
        self.weights=self.weights/weights_scale       
        self.bias=np.zeros((1,1,1,self.output_channels))

        if method == 'VALID':
            self.eta = np.zeros((self.shape[0], int((self.shape[1] - ksize + 1) / self.stride), int((self.shape[1] - ksize + 1) / self.stride),
             self.output_channels))

        if method == 'SAME':
            self.eta = np.zeros((self.shape[0], int(self.shape[1]/self.stride), int(self.shape[2]/self.stride),self.output_channels))

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

        if (self.shape[1] - ksize) % stride != 0:
            print ('input tensor width can\'t fit stride')
        if (self.shape[2] - ksize) % stride != 0:
            print ('input tensor height can\'t fit stride')

    def forward(self, x):
        self.shape=x.shape
        self.input_shape = self.shape
        self.batchsize = self.shape[0]

        if self.method == 'VALID':
            self.eta = np.zeros((self.shape[0], int((self.shape[1] - self.ksize + 1) / self.stride), int((self.shape[1] - self.ksize + 1) / self.stride),
             self.output_channels))

        if self.method == 'SAME':
            self.eta = np.zeros((self.shape[0], int(self.shape[1]/self.stride), int(self.shape[2]/self.stride),self.output_channels))

        self.output_shape = self.eta.shape
        
        col_weights = self.weights.reshape([-1, self.output_channels])
        if self.method == 'SAME':
            x = np.pad(x, (
                (0, 0), (int(self.ksize / 2), int(self.ksize / 2)), (int(self.ksize / 2), int(self.ksize / 2)), (0, 0)),
                             'constant', constant_values=0)

        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def gradient(self, eta):
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(self.eta, (
                (0, 0), (int(self.ksize / 2), int(self.ksize / 2)), (int(self.ksize / 2), int(self.ksize / 2)), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array([im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def backward(self, alpha=0.00001, lambd=0.0004):
        # lambd = L2 regularization
        self.weights *= (1 - alpha*lambd)
        self.bias *= (1 - alpha*lambd)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

class Relu(object):
    def __init__(self, shape):
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        self.x=np.zeros(x.shape)
        self.x = x
        return np.maximum(x, 0)

    def gradient(self, eta):
        self.eta = eta
        self.eta[self.x<0]=0
        return self.eta


class MaxPooling(object):
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.index = np.zeros(shape)
        self.output_shape = [shape[0], shape[1] / self.stride, shape[2] / self.stride, self.output_channels]

    def forward(self, x):
        self.output_channels=x.shape[-1]
        self.index=np.zeros(x.shape)
        out = np.zeros([x.shape[0], int(x.shape[1] / self.stride), int(x.shape[2] / self.stride), self.output_channels])

        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, int(i / self.stride), int(j / self.stride), c] = np.max(
                            x[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, i+int(index/self.stride), j + index % self.stride, c] = 1
        return out


    def gradient(self, eta):
        return np.repeat(np.repeat(eta, self.stride, axis=1), self.stride, axis=2) * self.index

def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col

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

def FC_forward(X,parameters):
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

    dW = np.dot(dZ,A_prev.T)/m+lambd*W
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

def FC_backward(AL,Y,lambd,caches):

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

    dA=grads['dA0']

    return grads,dA

def update_parameters(parameters,grads,learning_rate):

    L=len(parameters)//2

    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-grads["db"+str(l+1)]*learning_rate

    return parameters

def update_fparameters(fparameters,conv_grads,learning_rate):
    L=len(fparameters)//2
    for l in range(0,L):
        fparameters["W"+str(l+1)]=fparameters["W"+str(l+1)]-learning_rate*conv_grads["dW"+str(l+1)]
        fparameters["b"+str(l+1)]=fparameters["b"+str(l+1)]-learning_rate*conv_grads["db"+str(l+1)]

    return fparameters

def initialize_parameters(fc_layer):

    parameters = {}
    L=len(fc_layer)

    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(fc_layer[l],fc_layer[l-1])/np.sqrt(fc_layer[l-1])
        parameters["b"+str(l)]=np.zeros((fc_layer[l],1))

    return parameters

def model_train(X,Y,test_data,fc_layer,learning_rate=0.0075,epochs=10,mini_batch_size=1,lambd=0.7,print_cost=True):

    conv1=Conv_layer([mini_batch_size,28,28,1],6,5,1)
    relu1 = Relu(conv1.output_shape)
    pool1=MaxPooling(conv1.output_shape)

    conv2=Conv_layer(pool1.output_shape,16,5,1)
    relu2=Relu(conv2.output_shape)
    pool2=MaxPooling(conv2.output_shape)

    start=time.time()

    for i in range(0,epochs):
        np.random.seed(i)
        np.random.shuffle(X)
        np.random.seed(i)
        Y=Y.T
        np.random.shuffle(Y)
        Y=Y.T
        for j in range(0,Y.shape[1],mini_batch_size):
            mini_batchs=(X[j:j+mini_batch_size,:,:,:],Y[:,j:j+mini_batch_size])
            conv_out1=conv1.forward(mini_batchs[0])
            conv_out1=relu1.forward(conv_out1)
            A=pool1.forward(conv_out1)

            conv_out2=conv2.forward(A)
            conv_out2=relu2.forward(conv_out2)
            A=pool2.forward(conv_out2)

            A_flat=(A.reshape((A.shape[0],A.shape[1]*A.shape[2]*A.shape[3]))).T

            if i==0 and j==0:
                fc_layer.insert(0,A_flat.shape[0])
                parameters=initialize_parameters(fc_layer)

            AL,caches = FC_forward(A_flat,parameters)

            if j%1000==0:
                cross_entropy_cost=compute_cost(AL,mini_batchs[1])
                print("Cost after epoch {0} iterations {1} : {2} ".format(i,int(j/mini_batch_size),cross_entropy_cost))

            grads,dA = FC_backward(AL, mini_batchs[1], lambd,caches)

            dA=dA.T.reshape((A.shape[0],A.shape[1],A.shape[2],A.shape[3]))
            pool_grad2=pool2.gradient(dA)
            dZ2=relu2.gradient(pool_grad2)
            conv_grad2=conv2.gradient(dZ2)

            pool_grad1=pool1.gradient(conv_grad2)
            dZ1=relu1.gradient(pool_grad1)
            conv_grad1=conv1.gradient(dZ1)

            parameters = update_parameters(parameters, grads, learning_rate)
            conv2.backward(learning_rate, lambd)
            conv1.backward(learning_rate, lambd)

    end=time.time()
    print("Training time : %f s"%(end-start))

    if print_cost  :
        conv_out1=conv1.forward(X)
        conv_out1=relu1.forward(conv_out1)
        A=pool1.forward(conv_out1)

        conv_out2=conv2.forward(A)
        conv_out2=relu2.forward(conv_out2)
        A=pool2.forward(conv_out2)
        A_flat=(A.reshape((A.shape[0],A.shape[1]*A.shape[2]*A.shape[3]))).T
        AL,caches = FC_forward(A_flat,parameters)
        y=np.argmax(Y,axis=0)
        predictions= np.argmax(AL,axis=0)
        print ("Train accuracy : {}% ".format(((np.sum([ int(i) for i in y==predictions])/len(y))*100)))

        test_inputs,test_results=test_data
        test_x=test_inputs.reshape((10000,28,28,1))
        test_y=vectorized_result(test_results)


        conv_out1=conv1.forward(test_x[0:int(Y.shape[1]/5),:,:,:])
        conv_out1=relu1.forward(conv_out1)
        A=pool1.forward(conv_out1)

        conv_out2=conv2.forward(A)
        conv_out2=relu2.forward(conv_out2)
        A=pool2.forward(conv_out2)
        A_flat=(A.reshape((A.shape[0],A.shape[1]*A.shape[2]*A.shape[3]))).T
        AL,caches = FC_forward(A_flat,parameters)
        y=np.argmax(test_y[:,0:int(Y.shape[1]/5)],axis=0)
        predictions= np.argmax(AL,axis=0)
        print ("Test accuracy  : {}% ".format(((np.sum([ int(i) for i in y==predictions])/len(y))*100)))

    print("Learning rate : {0}   Batch size : {1}   Data size : {2}   Lambd : {3} Iterations : {4}".format(learning_rate,mini_batch_size,Y.shape[1],lambd,int(epochs*(Y.shape[1]/mini_batch_size))))
    print("Conv layer1 : {0}x{1}x{2}  mode: {3}".format(conv1.ksize,conv1.ksize,conv1.output_channels,conv1.method))
    print("Conv layer2 : {0}x{1}x{2}  mode: {3}".format(conv2.ksize,conv2.ksize,conv2.output_channels,conv2.method))
    print("Maxpooling layer : {0}x{1}".format(pool1.ksize,pool1.ksize))
    print("FullyConnect layer : {0}".format(fc_layer))




training_data,validation_data,test_data=load_data()
training_inputs,training_results=training_data
train_x=training_inputs.reshape((50000,28,28,1))
train_y=vectorized_result(training_results)


fc_layer=[80,10]
model_train(train_x[0:50000,:,:,:],train_y[:,0:50000],test_data,fc_layer,learning_rate=0.0062,epochs=1,mini_batch_size=5,lambd=0.0003,print_cost=True)
