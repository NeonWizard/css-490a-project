import keras, time, os
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam_v2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image
from keras import backend as K
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

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

# # -- Evaluation
# Accuracy over loss
# plt.plot(hist.history["accuracy"])
# plt.plot(hist.history['val_accuracy'])
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title("Model Accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Accuracy","Validation Accuracy","Loss","Validation Loss"])
# plt.savefig("graph_acc_loss.png")

def build_vgg16(num_classes):
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
  model.add(Dense(units=num_classes, activation="softmax"))

  return model

def normalize(self, x_train, x_test):
  # this function normalize inputs for zero mean and unit variance
  # it is used when training a model.
  # Input: training set and test set
  # Output: normalized training set and test set according to the trianing set statistics.
  mean = np.mean(x_train,axis=(0,1,2,3))
  std = np.std(x_train, axis=(0, 1, 2, 3))
  x_train = (x_train-mean)/(std+1e-7)
  x_test = (x_test-mean)/(std+1e-7)
  return x_train, x_test

def predict(model, x, batch_size=50):
  return model.predict(x, batch_size)

def train(model, train_data, test_data, learning_rate, epochs):
  # training parameters
  batch_size = 32

  # -- Compile model
  opt = adam_v2.Adam(learning_rate=learning_rate)
  model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy', f1_m, precision_m, recall_m])

  # model.summary()

  # -- Model fitting
  checkpoint = ModelCheckpoint("vgg16.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
  early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

  start_time = time.time()
  hist = model.fit(
    train_data,
    validation_data=test_data,
    # steps_per_epoch=len(train_data) // batch_size,
    epochs=epochs,
    # validation_steps=10,
    # batch_size=batch_size,
    callbacks=[checkpoint,early],
    verbose=2)

  print()
  print("--- Training Time ---")
  print("%s seconds" % round(time.time() - start_time, 3))
  print()

  return model, hist

def test(model, test_data):
  loss, acc, f1_score, precision, recall = model.evaluate(test_data)

  print("Loss: {}\nAcc: {}\nF1 Score: {}\nPrecision: {}\nRecall: {}".format(
    round(loss*100, 3),
    round(acc*100, 3),
    round(f1_score*100, 3),
    round(precision*100, 3),
    round(recall*100, 3),
  ))

  return loss, acc, f1_score, precision, recall

def predict(model, img):
  start_time = time.time()
  prediction_ind = model.predict(img).argmax(axis=-1)
  prediction = np.array(["paper", "rock", "scissors"])[prediction_ind][0]

  print()
  print("--- Single Image Inference Time ---")
  print("%s seconds" % round(time.time() - start_time, 3))
  print()

  print()
  print("Prediction: " + prediction)
  print()

  return prediction_ind

def prune(model, train_data, test_data, learning_rate, epochs, batch_size):
  end_step = np.ceil(1.0 * len(train_data) / batch_size).astype(np.int32) * epochs
  print("End step:", end_step)

  new_pruning_params = {
    'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                final_sparsity=0.90,
                                                begin_step=0,
                                                end_step=end_step)
                                                # frequency=100)
  }

  new_pruned_model = sparsity.prune_low_magnitude(model, **new_pruning_params)

  opt = adam_v2.Adam(learning_rate=learning_rate)
  new_pruned_model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy', f1_m, precision_m, recall_m])
  # new_pruned_model.summary()

  # -- Model fitting
  callbacks = [
    sparsity.UpdatePruningStep()
  ]

  start_time = time.time()
  hist = new_pruned_model.fit(
    train_data,
    validation_data=test_data,
    # steps_per_epoch=len(train_data) // batch_size,
    epochs=epochs,
    # validation_steps=10,
    # batch_size=batch_size,
    callbacks=callbacks)

  stripped_model = sparsity.strip_pruning(new_pruned_model)
  keras.models.save_model(stripped_model, "vgg16-pruned.h5")
  print("Saved pruned Keras model to vgg16-pruned.h5")
  # stripped_model.summary()

  return new_pruned_model, stripped_model


def main():
  should_train = False

  # -- load images
  id_generator = ImageDataGenerator(
      featurewise_center=False,  # set input mean to 0 over the dataset
      samplewise_center=False,  # set each sample mean to 0
      featurewise_std_normalization=False,  # divide inputs by std of the dataset
      samplewise_std_normalization=False,  # divide each input by its std
      zca_whitening=False,  # apply ZCA whitening
      rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
      width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
      height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
      horizontal_flip=True,  # randomly flip images
      vertical_flip=False)  # randomly flip images

  # TODO: consider using color_mode parameter to reduce to grayscale
  train_data = id_generator.flow_from_directory(
    directory="data/train", target_size=(224, 224))
  test_data = id_generator.flow_from_directory(
    directory="data/test", target_size=(224, 224))

  num_classes = 3
  learning_rate = 0.0001
  x_shape = [32, 32, 3]

  if should_train:
    model = build_vgg16(num_classes)
    model, hist = train(model, train_data, test_data, learning_rate, epochs=10)
  else:
    model = load_model("vgg16.h5", custom_objects={
      "f1_m": f1_m,
      "precision_m": precision_m,
      "recall_m": recall_m
    })

  pruned_model, stripped_model = prune(
    model,
    train_data,
    test_data,
    learning_rate,
    epochs=10,
    batch_size=32)

  # -- Evaluate model
  print()
  print("-- Testing BASELINE model --")
  test(model, test_data)
  print()

  print()
  print("-- Testing PRUNED model --")
  test(pruned_model, test_data)
  print()

  # -- Classify a single image
  img = image.load_img("data/validate/rock1.png", target_size=(224,224))
  img = np.asarray(img)
  img = np.expand_dims(img, axis=0)

  predict(model, img)
  predict(pruned_model, img)

  # -- Measure pruning size difference
  before_size = os.path.getsize("vgg16.h5")
  after_size = os.path.getsize("vgg16-pruned.h5")
  print("Size of model BEFORE pruning:")
  print(round(before_size / float(2**20), 2), "MB")

  print("Size of model AFTER pruning:")
  print(round(after_size / float(2**20), 2), "MB")

  print("Size percentage of original:")
  print("{}%".format(round(before_size/after_size * 100, 2)))

  # -- Convert and save TFLite models
  base_tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
  pruned_tflite_model = tf.lite.TFLiteConverter.from_keras_model(stripped_model).convert()

  with open("vgg16.tflite", 'wb') as f:
    f.write(base_tflite_model)

  print("Saved base TFLite model to 'vgg16.tflite'.")

  with open("vgg16-pruned.tflite", 'wb') as f:
    f.write(pruned_tflite_model)

  print("Saved pruned TFLite model to 'vgg16-pruned.tflite'.")

if __name__ == "__main__":
  main()