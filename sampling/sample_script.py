import numpy as np
import scipy as sci
from scipy import ndimage
import matplotlib.pyplot as plt

import ristretto.mf
from ristretto.mf import *

# Read in image
A = sci.ndimage.imread('/home/ben/Dropbox/Shared Projects/CNN/imagenet/cropped_panda.jpg', flatten=True)

# Display image
fig = plt.figure(facecolor="white", figsize=(6.5, 7.5), edgecolor='k')
plt.imshow(A, cmap = 'gray')
plt.axis('off')
plt.show()

# Print shape
m,n = A.shape
print('Dimensions:', A.shape)


#
# Deterministic column ID
#

# Change value for k, maybe use a loop
C, V = interp_decomp(A, k=80, mode='column', index_set=True)

# Display image
fig2 = plt.figure(facecolor="white", figsize=(6.5, 7.5), edgecolor='k')
partial_image = np.zeros(A.shape)
partial_image[:,C] = A[:,C]
plt.imshow(partial_image, cmap = 'gray')
plt.axis('off')
plt.show()

sci.misc.imsave('/home/ben/Dropbox/Shared Projects/CNN/imagenet/cropped_panda_id.jpg', partial_image)

#
# Deterministic row ID
#
V, R = interp_decomp(A, k=99, mode='row', index_set=True)

# Display image
fig2 = plt.figure(facecolor="white", figsize=(6.5, 7.5), edgecolor='k')
partial_image = np.zeros(A.shape)
partial_image[R,:] = A[R,:]
plt.imshow(partial_image, cmap = 'gray')
plt.axis('off')
plt.show()


sci.misc.imsave('/home/ben/Dropbox/Shared Projects/CNN/imagenet/cropped_panda_id.jpg', partial_image)