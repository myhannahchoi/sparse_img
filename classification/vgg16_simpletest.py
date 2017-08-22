from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import scipy as sci
from scipy import ndimage
from ristretto.mf import *
import pdb
from keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# image file
img_path = '/home/wangnxr/Documents/images/car.png'
# Random sampling?
rand=False
# model
model = ResNet50(weights='imagenet')

# Colour image
img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)

# Grayscale image for ID
x1 = sci.ndimage.imread(img_path, flatten=True)
x = sci.misc.imresize(x1, (224,224), mode='F')

detection_score = np.zeros(300)

for k_slct in range(1,225):
    print "k=%i" % k_slct

    if rand:
        C = np.arange(224)
        np.random.shuffle(C)
        C = C[:k_slct]
    else:
        C, V = interp_decomp(x, k=k_slct, mode='column', index_set=True)

    partial_image = np.random.rand(*img.shape) * 256
    partial_image[:,:,C] = img[:,:,C]
    #sci.misc.imsave('/home/wangnxr/test.png', np.transpose(partial_image, (1,2,0)))
    partial_img = np.expand_dims(partial_image, axis=0)
    partial_img = preprocess_input(partial_img)

    preds = model.predict(partial_img)
    #record class probability
    detection_score[k_slct]=preds[0,436]

#Plot detection curve
plt.plot(detection_score)
plt.title("Car detection score vs. K")
plt.xlabel("K")
plt.ylabel("Class probability")
plt.savefig("/home/wangnxr/Documents/simp_imgs_proj/results/car_cur.png")
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=5)[0])
# Save the partial image for inspection
sci.misc.imsave('/home/wangnxr/test.png', np.transpose(partial_image, (1,2,0)))
