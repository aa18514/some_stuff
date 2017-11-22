import h5py
import caffe


if __name__ == "__main__": 
    lenet = caffe.Net("lenet.prototxt", "lenet_iter_10000.caffemodel", caffe.TEST)
    hf = h5py.File('fc.h5', 'w')
    hf.create_dataset('fc1_weights', data = lenet.params['ip1'][0].data)
    hf.create_dataset('fc1_biases', data = lenet.params['ip2'][1].data)
    hf.create_dataset('fc2_weights', data = lenet.params['ip2'][0].data)
    hf.create_dataset('fc2_biases', data = lenet.params['ip2'][1].data)
    hf.close()
