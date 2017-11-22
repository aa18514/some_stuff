

class conv():

        def __init__(self, outputChannels, inputChannels, kernelDimension, imageDimension, stride): 
            self.__outputChannels = outputChannels
            self.__inputChannels = inputChannels
            self.__kernelDimension = kernelDimension
            self.__imageDimension = imageDimension
            self.__stride = stride
        
        def get_outputchannels(self): 
            return self.__outputChannels

        def get_inputchannels(self): 
            return self.__inputChannels

        def get_kernelDimension(self):
            return self.__kernelDimension
        
        def get_stride(self):
            return self.__stride
       
        def get_imageDimension(self): 
            return self.__imageDimension
