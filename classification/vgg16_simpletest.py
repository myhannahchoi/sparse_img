from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

from ristretto.mf import *

model = VGG16(weights='imagenet')

img_path = 'panda.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
# Change value for k, maybe use a loop
C, V = interp_decomp(x, k=80, mode='column', index_set=True)

# Display image

partial_image = np.zeros(x.shape)
partial_image[:,C] = x[:,C]

x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)