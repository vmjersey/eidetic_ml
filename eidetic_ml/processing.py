import numpy as np
import sys
import subprocess
import pkg_resources

#Packages that can help speed up program
warn_packages = {'numba'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = warn_packages - installed

if missing:
    print("The following packages are not installed: ", missing)


from numba.experimental import jitclass
from numba import int32, float32    # import the types
spec = [
    ('lookback', int32),               # a simple scalar field
    ('lstmdataset', float32[:]),       # an array field
    ('dataset', float32[:]),
]


    
#@jitclass(spec)
class LSTM_DATA:
    """ 
        Basic Class for handling LSTM data.  


        transform(lstmdataset,dataset):
            Transforms a 2d into a 3d dataset of dimension of the type
            (number of samples,lookback, number of features)
    """

    def __init__(self,lookback):
        self.lookback = lookback

    def transform(self,lstmdataset,dataset):
        num_samples = dataset.shape[0]
        num_features = dataset.shape[1]
        for i in range(0, num_samples - self.lookback, 1):
            sequence = dataset[i:i + self.lookback]
            lstmdataset[i] = sequence



class Encoding:

    """The Encoding Class contains functions for encoding and decoding images.
    Methods
    -----
    decode_rle_mask(encoded_pixels,height,width)
        Takes in an rle encoded mask and returns the decoded mask.
    
    """


    def decode_rle_mask(rle_string,height,width):
        """Takes in an rle encoded mask and returns the mask.
        
        Parameters
        ----------
        encoded_pixels : str, required
            A string containing the run length encoded mask
        height: int, required
        width: int, required

        """    

    
        if rle_string == None:
            return np.zeros((height, width))
        else:
            rle_numbers = []
            for num_string in rle_string.split(' '):
                rle_numbers.append(int(num_string))

            rle_pairs = np.array(rle_numbers).reshape(-1,2)
            img = np.zeros(height*width, dtype=np.uint8)
            for index, length in rle_pairs:
                index -= 1
                img[index:index+length] = 255
            img = img.reshape(width,height)
            img = img.T
            mask = np.stack((img,)*3, axis=-1)
            return mask



