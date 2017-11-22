import h5py
import caffe


if __name__ == "__main__": 
    lenet = caffe.Net("lenet.prototxt", "lenet_iter_10000.caffemodel", caffe.TEST)
    hf = h5py.File('conv.h5', 'w')
    hf.create_dataset('conv1_weights', data = lenet.params['conv1'][0].data)
    hf.create_dataset('conv1_biases', data = lenet.params['conv1'][1].data)
    hf.create_dataset('conv2_weights', data = lenet.params['conv2'][0].data)
    hf.create_dataset('conv2_biases', data = lenet.params['conv2'][1].data)
    hf.close()
