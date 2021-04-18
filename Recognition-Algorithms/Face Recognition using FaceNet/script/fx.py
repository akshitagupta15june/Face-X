import numpy as np

def prewhiten(x):
    if x.ndim==4:
        axis=(1, 2, 3)
        size=x[0].size
    elif x.ndim==3:
        axis=(0, 1, 2)
        size=x.size
    else:
        raise ValueError('Dimension should be 3 or 4.')

    mean=np.mean(x,axis=axis,keepdims=True)
    std=np.std(x,axis=axis,keepdims=True)
    std_adj=np.maximum(std,1.0/np.sqrt(size))
    y=(x-mean)/std_adj
    return y


def l2_normalize(x):
    if x.ndim==2:
        temp=[]
        for i in range(x.shape[0]):
            temp.append(l2_normalize(x[i,:]))
        return np.array(temp)
    elif x.ndim==1:
        return x/np.sqrt(np.sum(np.multiply(x, x)))
