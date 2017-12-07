
# coding: utf-8

# In[1]:


import tensorflow as tf; print(tf.__version__)


# In[1]:


import keras
from timeit import default_timer as timer
from keras import optimizers
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard

use_sfc_schedule  = False
use_cifar         = False # False = use MNIST

batch_size        = 128
epochs            = 100
iterations        = 391
num_classes       = 10
log_filepath      = './lenet'

def build_model():
    input_shape = (32, 32, 3) if use_cifar else (28, 28, 1)
    
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def scheduler(epoch):
    if use_sfc_schedule:
        if epoch < epochs * 0.6:
            return 0.1
        if epoch < epochs * 0.8:
            return 0.1*.2
        if epoch < epochs * 0.9:
            return 0.1*.2*.2
        else:
            return 0.1*.2*.2*.2
    else:
        return 0.1

    
def learn_model():
    # load data
    if use_cifar:
        (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = cifar10.load_data()
    else:
        (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = mnist.load_data()
        x_train_orig = x_train_orig.reshape(-1, 28, 28, 1)
        x_test_orig = x_test_orig.reshape(-1, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train_orig, num_classes)
    y_test = keras.utils.to_categorical(y_test_orig, num_classes)
    x_train = x_train_orig.astype('float32')
    x_test = x_test_orig.astype('float32')
    x_train /= 255
    x_test /= 255

    # build network
    model = build_model()
    print(model.summary())
    # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

    # start traing 
    start = timer()
    model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,callbacks=cbks,
                  validation_data=(x_test, y_test), shuffle=True)
    end = timer()
    print("Training time:", (end - start))
    # save model
    model.save('lenet.h5')


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

def pop_cnn_layer(model):
    pop_layer(model)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def append_svm():
    model = pop_cnn_layer(model)
    svm_trainset = model.predict(x_train)
    svm_testset = model.predict(x_test)

    predictor = svm.SVC(kernel="rbf")

    start = timer()
    predictor.fit(svm_trainset, y_train_orig)
    end = timer()
    print("Training time:", (end - start))

    svm_predicted_y = predictor.predict(svm_testset)

    total = 0
    success = 0
    for i, y in enumerate(svm_predicted_y):
        total += 1
        success += 1 if y == y_test_orig[i] else 0

    print(1.*success/total)


if __name__ == '__main__':
    learn_model()

    # Optional step if last layer should be dropped and SVM should be added to the top
    #append_svm()