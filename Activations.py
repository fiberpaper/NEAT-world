## v0 - 11/06/2017
## activations

import sys
import numpy as np

# Try to import cupy to speed up processing
try :
    import cupy as cp
except :
    print("Cupy not available")

if ('cupy' in sys.modules):
    cupy_ready=True
else:
    cupy_ready=False

# Based on tests on current graphic card (float64):
# -> the minimum size for a gpu array to run faster sigmoid is 30 000
# -> the minimum size for a gpu array to run faster softplus is 30 000
# -> etc.
# -> the maximum size for a gpu array to run (memory limit) is 100 000 000

gpu_array_max_size=100000000

# sigmoid
def sigmoid_ac(z):
    if (cupy_ready and (z.size>30000) and (z.size<=gpu_array_max_size)):
        z_gpu=cp.asarray(z)
        r_gpu=1.0 / (1.0 + cp.exp(-z_gpu))
        return cp.asnumpy(r_gpu)
    else:
        return 1.0 / (1.0 + np.exp(-z))


# softplus
def softplus_ac(z):
    if (cupy_ready and (z.size>30000) and (z.size<=gpu_array_max_size)):
        z_gpu=cp.asarray(z)
        r_gpu=0.2 * cp.log(1 + cp.exp(z_gpu))
        return cp.asnumpy(r_gpu)
    else:
        return 0.2 * np.log(1 + np.exp(z))

# relu
def relu_ac(z):
    return np.where(z>=0.0, z, 0.0)

# elu
def elu_ac(z):
    alpha=1.0
    if (cupy_ready and (z.size>30000) and (z.size<=gpu_array_max_size)):
        z_gpu=cp.asarray(z)
        r_gpu=np.where(z_gpu>=0.0, z_gpu, alpha*np.exp(z_gpu)-1)
        return cp.asnumpy(r_gpu)
    else:
        return np.where(z>=0.0, z, alpha*np.exp(z)-1)

# selu
def selu_ac(z):
    alpha=1.6732632423543772848170429916717
    scale=1.0507009873554804934193349852946
    if (cupy_ready and (z.size>30000) and (z.size<=gpu_array_max_size)):
        z_gpu=cp.asarray(z)
        r_gpu=scale*cp.where(z_gpu>=0.0, z_gpu, alpha*cp.exp(z_gpu)-alpha)
        return cp.asnumpy(r_gpu)
    else:
        return scale*np.where(z>=0.0, z, alpha*np.exp(z)-alpha)

# hat
def hat_ac(z):
    return np.maximum(0.0, 1 - np.abs(z))

# sin
def sin_ac(z):
    if (cupy_ready and (z.size>40000) and (z.size<=gpu_array_max_size)):
        z_gpu=cp.asarray(z)
        r_gpu=cp.sin(z_gpu)
        return cp.asnumpy(r_gpu)
    else:
        return np.sin(z)

# tanh
def tanh_ac(z):
    if (cupy_ready and (z.size>400000) and (z.size<=gpu_array_max_size)):
        z_gpu=cp.asarray(z)
        r_gpu=cp.tanh(z_gpu)
        return cp.asnumpy(r_gpu)
    else:
        return np.tanh(z)

# absolute
def abs_ac(z):
    return np.abs(z)

# identity
def id_ac(z):
    return z

# gauss
def gauss_ac(z):
    if (cupy_ready and (z.size>20000) and (z.size<=gpu_array_max_size)):
        z_gpu=cp.asarray(z)
        r_gpu=cp.exp(-5.0 * z_gpu**2)
        return cp.asnumpy(r_gpu)
    else:
        return np.exp(-5.0 * z**2)
    
# activation class
class activations:

    def __init__(self,keyword):

        self.list=list()
        self.op=list()

        if (keyword=='standard' or keyword=='sigmoid'):

            self.list.append('sigmoid')
            self.op.append(sigmoid_ac)

        elif (keyword=='relu'):

            self.list.append('relu')
            self.op.append(relu_ac)
            
        elif (keyword=='gauss'):

            self.list.append('gauss')
            self.op.append(relu_ac)

        elif (keyword=='elu'):

            self.list.append('elu')
            self.op.append(elu_ac)

        elif (keyword=='selu'):

            self.list.append('selu')
            self.op.append(selu_ac)

        elif (keyword=='all'):

            self.list.append('sigmoid')
            self.op.append(sigmoid_ac)
            self.list.append('relu')
            self.op.append(relu_ac)
            self.list.append('elu')
            self.op.append(elu_ac)
            self.list.append('selu')
            self.op.append(selu_ac)
            self.list.append('gauss')
            self.op.append(gauss_ac)
            self.list.append('id')
            self.op.append(id_ac)
            self.list.append('abs')
            self.op.append(abs_ac)
            self.list.append('tanh')
            self.op.append(tanh_ac)
            self.list.append('sin')
            self.op.append(sin_ac)
            self.list.append('hat')
            self.op.append(hat_ac)
            self.list.append('softplus')
            self.op.append(softplus_ac)

        else:
            
            raise ValueError('Activation list unknown')
