import tensorflow as tf
import h5py
import caffe_pb2
import numpy as np 
import datetime
import conv
from google.protobuf import text_format
import tensorflow

def quantize(m1, m2, imageDimension): 
    m2 = m2.flatten()
    completeSlidingWindow = []
    for i in range(2, imageDimension - 2): 
        for j in range(2, imageDimension - 2): 
            sliding_window = np.array([])
            cols = m1[:,j-2 : j+3]
            for row in range(-2, 3):
                sliding_window = np.append(sliding_window, cols[i + row])
            completeSlidingWindow.append(sliding_window)
    a = [np.dot(x, m2) for x in completeSlidingWindow]
    return np.array(a).reshape(imageDimension - 4, imageDimension - 4)

def maxpool(m, image): 
    result = [] 
    for i in range(0, image, 2): 
        for j in range(0, image, 2): 
            maxpool_vals = []
            a = m[:,j:j+2]
            maxpool_vals.append(a[i])
            maxpool_vals.append(a[i+1])
            result.append(max(np.array(maxpool_vals).flatten()))
    return np.array(result).reshape(int(image * 0.5), int(image * 0.5))

def tf_conv_cpu(matrix_1, conv, pool, data, conv_1_weights, conv_2_weights, conv1_bias, conv2_bias, ip1_weights, ip2_weights, ip1_bias, ip2_bias):
    StartTime = datetime.datetime.now()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    inputChannels = [data.input_param.shape[0].dim[1], conv[0].convolution_param.num_output]
    A = tf.nn.conv2d(matrix_1.reshape(10000, 28, 28, 1), filter = weights.reshape(5, 5, 1, 20), strides = [1, 1, 1, 1], padding = "VALID") 
    B = tf.nn.max_pool(A, [1, 2, 2, 1], [1, 2, 2, 1],  padding = "SAME") + conv1_bias
    C = tf.nn.conv2d(B, filter = weights2.reshape(5, 5, 20, 50), strides = [1, 1, 1, 1],  padding = "VALID")
    relu = tf.nn.max_pool(C, [1, 2, 2, 1], [1, 2, 2, 1], padding = "SAME")
    a = sess.run(relu)
    b = a.transpose().transpose((3, 0, 2, 1)) + conv2_bias.reshape(1, 50, 1, 1)
    Z = tf.matmul(ip1_weights.reshape(500, 800), b.reshape(10000, 800).T) + ip1_bias.reshape(500, 1)
    Y = tf.matmul(ip2_weights.reshape(10, 500), tf.nn.relu(Z)) + ip2_bias.reshape(10, 1)
    relu = sess.run(Y)
    sess.close()
    FinalTime = datetime.datetime.now()
    print("tensorflow took: %fs" % (10000 * ((40 * 24 * 24 * 5 * 5) + (24 * 24 * 20) + (20 * 12 * 12 * 4) +  (2 * 12 * 12 * 5 * 5 * 20 * 50) + (50 * 20 * 12 * 12) + (50 * 12 * 12) + (50 * 8 * 8 * 4) +  (500 * 4 * 800) + (10 * 500 * 2))/(FinalTime - StartTime).total_seconds()))
    print((FinalTime - StartTime).total_seconds())
    print(relu)
    return relu

def numpy_conv(net, matrix_1): 
    initialTime = datetime.datetime.now()
    conv_weights = net.params['conv1'][0].data.reshape(1, 20, 5, 5)
    bias = net.params['conv1'][1].data.reshape(20)
    c = []
    m = []
    for i in range(20): 
        c.append((maxpool(quantize(matrix_1, conv_weights[0][i], 28) + bias[i], 24)))
    conv_weights = net.params['conv2'][0].data.reshape(50, 20, 5, 5)
    bias = net.params['conv2'][1].data.reshape(50)
    secondlayerfinal = []
    for i in range(50): 
        secondlayerconv = []
        for j in range(20): 
            secondlayerconv.append(quantize(c[j], conv_weights[i][j], 12))
        secondlayerfinal.append(sum(secondlayerconv) + bias[i])
    secondlayerfinal = np.array(secondlayerfinal).reshape(1, 50, 8, 8)
    secondlayermaxpool = []
    for i in range(50): 
        secondlayermaxpool.append(maxpool(secondlayerfinal[0][i], 8))
    secondlayermaxpool = np.array(secondlayermaxpool).reshape(1, 50, 4, 4)
    secondlayermaxpool = secondlayermaxpool.flatten()
    secondlayermaxpool = secondlayermaxpool.reshape(1, 800)
    fc1 = net.params['ip1'][0].data.reshape(500, 800)
    c = np.dot(secondlayermaxpool, fc1.T) + net.params['ip1'][1].data.reshape(500)
    c[c < 0.0] = 0.0
    fc2 = net.params['ip2'][0].data.reshape(10, 500) 
    fc2o = np.dot(c.flatten(), fc2.T) + net.params['ip2'][1].data.reshape(10)
    finalTime = datetime.datetime.now()
    print("numpy took: %fs"% (finalTime - initialTime).total_seconds())
    print(fc2o)

if __name__ == "__main__":
    lenet = caffe_pb2.NetParameter()
    text_format.Merge(open("lenet.prototxt").read(), lenet)
    lenet.MergeFromString(open("lenet_iter_10000.caffemodel", "rb").read())
    conv = []
    pool = []
    data = None
    layers = lenet.layer
    for layer in layers: 
        if(layer.type == "Input"):
            data = layer
        if(layer.type == "Convolution"): 
            conv.append(layer)
        elif(layer.type == "Pooling"): 
            pool.append(layer)
    hf = h5py.File('conv.h5', 'r')
    n1 = hf.get('conv1_weights')
    n2 = hf.get('conv2_weights')
    conv1_weights = np.array(n1).transpose((2, 3, 1, 0))
    conv2_weights = np.array(n2).transpose((2, 3, 1, 0))
    conv1_bias = np.array(hf.get('conv1_biases'))
    conv2_bias = np.array(hf.get('conv2_biases'))
    hf.close()
    hf = h5py.File('fc.h5', 'r')
    fc1_weights = np.array(hf.get('fc1_weights'))
    fc1_bias = np.array(hf.get('fc1_biases'))
    fc2_weights = np.array(hf.get('fc2_weights'))
    fc2_bias = np.array(hf.get('fc2_biases'))
    #net = caffe.Net("lenet.prototxt","lenet_iter_10000.caffemodel", caffe.TEST) 
    matrix_1 = np.random.randint( 0, 256, (10000, 28, 28))
    matrix_1 = np.array(matrix_1, np.float32)
    print(matrix_1.shape)
    tf_conv_cpu( matrix_1, conv, pool, data, conv1_weights, conv2_weights, conv1_bias, conv2_bias, fc1_weights, fc2_weights, fc1_bias, fc2_bias)
    print("HELLO")
    print(matrix_1[0][0].shape)
    matrix_2 = matrix_1[1]
    matrix_2 = matrix_2[np.newaxis,np.newaxis,:,:]
    net.blobs['data'].reshape(*matrix_2.shape)
    net.blobs['data'].data[...] = matrix_2
    net.forward()
    print(net.blobs['ip2'].data.flatten())
    layers = lenet.layer
    for layer in layers:
        print(layer.type)
        if(layer.type == "Convolution"):
            kernel = layer.convolution_param.kernel_size[0]
            stride = layer.convolution_param.stride[0]
            print(kernel)
       
    print(layer.convolution_param.num_output)
