import sys
import pickle
import scipy.misc
import os.path
from imageio import imwrite, imread
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det


# Functions to read and write Images
def readImages(image_path):
    img = imread(image_path)
    reshape_value = 1
    for i in img.shape:
        reshape_value *= i
    return img.reshape((1, reshape_value)), img.shape


class Hill:
    def __init__(self, data, file_name, key_path=None):
        self.data = data
        self.chunk = self.computer_chunk()
        file_name = file_name + '.key'
        if os.path.isfile(file_name):
            self._key = pickle.load(open(file_name, "rb"))
            print('Using the ' + file_name)
        else:
            self._key = np.random.random_integers(
                0, 100, (self.chunk, self.chunk))
            if det(self._key) == 0:
                self._key = np.random.random_integers(
                    0, 100, (self.chunk, self.chunk))
            pickle.dump(self._key, open(file_name, "wb"))
        print(self._key.dtype)
        print(self._key.shape)
        print(self._key)
        self.reversed_key = np.matrix(self._key).I.A
        print(self.reversed_key.dtype)
        print(self.reversed_key.shape)
        print(self.reversed_key)

    def computer_chunk(self):
        max_chunk = 100
        data_shape = self.data.shape[1]
        print(data_shape)

        for i in range(max_chunk, 0, -1):
            if data_shape % i == 0:
                return i

    @property
    def key(self):
        return self._key

    def encode(self, data):
        crypted = []
        chunk = self.chunk
        key = self._key
        for i in range(0, len(data), chunk):
            temp = list(np.dot(key, data[i:i + chunk]))
            crypted.append(temp)
        crypted = (np.array(crypted)).reshape((1, len(data)))
        return crypted[0]

    def decode(self, data):
        uncrypted = []
        chunk = self.chunk
        reversed_key = self.reversed_key
        for i in range(0, len(data), chunk):
            temp = list(np.dot(reversed_key, data[i:i + chunk]))
            uncrypted.append(temp)
        uncrypted = (np.array(uncrypted)).reshape((1, len(data)))
        return uncrypted[0]


# Start of program
if len(sys.argv) > 1:
    imageInputName = sys.argv[1]
else:
    raise Exception('Please enter image correct image path')
try:
    img, original_shape = readImages(imageInputName)
except:
    raise Exception('No Image found in your provided path')

img, original_shape = readImages(imageInputName)
hill = Hill(data=img, file_name=imageInputName)

# Testing zone
print(img.shape)


# Now Encoding image
encodedImageVector = hill.encode(img[0])
encryptedImage = encodedImageVector.reshape(original_shape)
imageName = imageInputName.split('.')[0]
img_extension = imageInputName.split('.')[1]
encryptedImageName = '{0}-encoded.{1}'.format(imageName, img_extension)
encryptedImage = encryptedImage.astype('uint8')
imwrite(encryptedImageName, encryptedImage)
pickle.dump(encodedImageVector, open(encryptedImageName + '.pk', "wb"))

input("Press Enter")


# Now decoding the image
imageVector = pickle.load(open(encryptedImageName + '.pk', 'rb'))
decryptedImageVector = hill.decode(imageVector)
decryptedImage = decryptedImageVector.reshape(original_shape)
decryptedImageName = '{0}-decoded.{1}'.format(imageName, img_extension)

# Save the image
imwrite(decryptedImageName, decryptedImage)
