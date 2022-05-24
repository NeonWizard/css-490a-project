import keras, time
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam_v2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image
from keras import backend as K
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

# -- load images
id_generator = ImageDataGenerator()
# TODO: consider using color_mode parameter to reduce to grayscale
train_data = id_generator.flow_from_directory(
  directory="data/train", target_size=(224, 224))
test_data = id_generator.flow_from_directory(
  directory="data/test", target_size=(224, 224))

# -- build model
model = Sequential()

# 2x convolution layer of 64 channel of 3x3 kernal and same padding
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

# 1x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# 2x convolution layer of 128 channel of 3x3 kernal and same padding
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

# 1x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# 3x convolution layer of 256 channel of 3x3 kernal and same padding
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

# 1x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# 3x convolution layer of 512 channel of 3x3 kernal and same padding
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# 1x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# 3x convolution layer of 512 channel of 3x3 kernal and same padding
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

# 1x maxpool layer of 2x2 pool size and stride 2x2
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=3, activation="softmax"))

# -- Evaluation metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# -- Compile model
opt = adam_v2.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy', f1_m, precision_m, recall_m])

model.summary()

# -- Model fitting
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

start_time = time.time()
hist = model.fit(
  train_data,
  # steps_per_epoch=100,
  validation_data=test_data,
  # validation_steps=10,
  epochs=10,
  batch_size=8,
  callbacks=[checkpoint,early])
print()
print("--- Training Time ---")
print("%s seconds" % round(time.time() - start_time, 3))
print()

# # -- Evaluation
# Accuracy over loss
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","Loss","Validation Loss"])
plt.savefig("graph_acc_loss.png")

# -- testing
saved_model = load_model("vgg16_1.h5", custom_objects={
  "f1_m": f1_m,
  "precision_m": precision_m,
  "recall_m": recall_m
})

loss, acc, f1_score, precision, recall = saved_model.evaluate(test_data)
print()
print("-- Evaluation on testing dataset --")
print("Loss: {}\nAcc: {}\nF1 Score: {}\nPrecision: {}\nRecall: {}".format(
  round(loss*100, 3),
  round(acc*100, 3),
  round(f1_score*100, 3),
  round(precision*100, 3),
  round(recall*100, 3),
))
print()

img = image.load_img("data/validate/rock1.png", target_size=(224,224))
img = np.asarray(img)
img = np.expand_dims(img, axis=0)

start_time = time.time()
prediction_ind = saved_model.predict(img).argmax(axis=-1)
prediction = np.array(["paper", "rock", "scissors"])[prediction_ind][0]

print()
print("--- Single Image Inference Time ---")
print("%s seconds" % round(time.time() - start_time, 3))
print()

print()
print("Prediction: " + prediction)
print()
