from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

model = VGG16(weights='imagenet')

img_path = 'images/dog.png'

#There is an interpolation method to match the source size with the target size
#image loaded in PIL (Python Imaging Library)
img = image.load_img(img_path,color_mode='rgb', target_size=(224, 224))

# Converts a PIL Image to 3D Numpy Array
x = image.img_to_array(img)
# Adding the fouth dimension, for number of images
x = np.expand_dims(x, axis=0)

# mean centering with respect to Image
x = preprocess_input(x)
features = model.predict(x)
p = decode_predictions(features)

for x in p[0]:
  print(x[1], ": ", x[2])