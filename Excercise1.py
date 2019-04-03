import numpy as np

def sigmoid(x):
    
    s=1/(1+np.exp(-x))
    
    return s

x=np.array([1,2,3])
sigmoid(x)



def sigmoid_derivative(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)
    return ds

x=np.array([1,2,3])
print("sigmoid_derivative(x)="+str(sigmoid_derivative(x)))

def image2vector(image):
    v=image.reshape((image.shape[0]*image.shape[1]*image.shape[2],1))
    return v

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print("image2vector(image)="+str(image2vector(image)))

def normalizeRows(x):
    x_norm=np.linalg.norm(x,axis=1,keepdims=True)
    x=x/x_norm
    return x

x=np.array([[0,3,4],[1,6,4]])
print("normalizeRows(x)="+str(normalizeRows(x)))

def softMax(x):
    x_exp=np.exp(x)
    x_sum=x_exp.sum(axis=1,keepdims=True)
    s=x_exp/x_sum
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softMax(x) ="+str(softMax(x)))

def L1(yhat,y):
    loss=(np.abs(y-yhat)).sum()
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1(yhat,y)="+str(L1(yhat,y)))

def L2(yhat,y):
    loss=(np.dot(y-yhat,y-yhat)).sum()
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2(yhat,y)="+str(L2(yhat,y)))




