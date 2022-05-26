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

class MyVGG16:
  def __init__(self, train=True):
    self.num_classes = 3
    self.weight_decay = 0.0005
    self.x_shape = [32, 32, 3]

    if train:
      self.model = self.build_model()
      self.train()
    else:
      self.model = load_model("vgg16.h5", custom_objects={
        "f1_m": f1_m,
        "precision_m": precision_m,
        "recall_m": recall_m
      })

  def build_model(self):
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
    model.add(Dense(units=self.num_classes, activation="softmax"))

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

  # def normalize_production(self, x):
  #   # this function is used to normalize instances in production according to saved training set statistics
  #   # Input: X - a training set
  #   # Output X - a normalized training set according to normalization constants.

  #   # these values produced during first training and are general for the standard cifar10 training set normalization
  #   mean = 120.707
  #   std = 64.15
  #   return (x-mean)/(std+1e-7)

  def predict(self, x, normalize=True, batch_size=50):
    if normalize:
      x = self.normalize_production(x)

    return self.model.predict(x,batch_size)

  def train(self):
    # training parameters
    batch_size = 32
    max_epochs = 10
    learning_rate = 0.0001

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

    # train_data = train_data.astype("float32")
    # test_data = test_data.astype("float32")
    # train_data, test_data = self.normalize(train_data, test_data)

    # -- Compile model
    opt = adam_v2.Adam(learning_rate=learning_rate)
    self.model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy', f1_m, precision_m, recall_m])

    self.model.summary()

    # -- Model fitting
    checkpoint = ModelCheckpoint("vgg16.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10, verbose=1, mode='auto')

    start_time = time.time()
    hist = self.model.fit(
      train_data,
      validation_data=test_data,
      # steps_per_epoch=len(train_data) // batch_size,
      epochs=max_epochs,
      # validation_steps=10,
      # batch_size=batch_size,
      callbacks=[checkpoint,early],
      verbose=2)

    print()
    print("--- Training Time ---")
    print("%s seconds" % round(time.time() - start_time, 3))
    print()

    return self.model

  def test(self, test_data):
    loss, acc, f1_score, precision, recall = self.model.evaluate(test_data)

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

  def predict(self, img):
    start_time = time.time()
    prediction_ind = self.model.predict(img).argmax(axis=-1)
    prediction = np.array(["paper", "rock", "scissors"])[prediction_ind][0]

    print()
    print("--- Single Image Inference Time ---")
    print("%s seconds" % round(time.time() - start_time, 3))
    print()

    print()
    print("Prediction: " + prediction)
    print()

    return prediction_ind


def main():
  model = MyVGG16(train=True)

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
  test_data = id_generator.flow_from_directory(
  directory="data/test", target_size=(224, 224))
  model.test(test_data)

  img = image.load_img("data/validate/rock1.png", target_size=(224,224))
  img = np.asarray(img)
  img = np.expand_dims(img, axis=0)

  model.predict(img)

if __name__ == "__main__":
  main()