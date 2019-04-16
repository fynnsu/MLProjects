from keras.applications.densenet import DenseNet121

from keras.layers import Dense, Activation, Flatten, Dropout
from keras import backend as K

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.densenet import preprocess_input
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

import stl10_input as stl
import numpy as np
import pandas as pd

def build_fine_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)

    fine_model = Model(inputs=base_model.input, outputs=predictions)
    
    return fine_model

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

images = stl.read_all_images('./data/stl10_binary/train_X.bin')

train_generator = train_datagen.flow_from_directory('./img', target_size=(96, 96), batch_size=64)

fc_layers = [64, 16]
num_classes = 10
adam = Adam(lr=0.00001)

model = build_fine_model(base_model, 0.2, fc_layers, num_classes)

model.load_weights("./checkpoints/" + 'test' + "_model_weights.h5")

model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

def lr_decay(epoch):
    if epoch%1 == 0 and epoch!=0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr/2)
        print("LR changed to {}".format(lr/2))
    return K.get_value(model.optimizer.lr)

learning_rate_schedule = LearningRateScheduler(lr_decay)

filepath = './checkpoints/' + 'test' + '_model_weights.h5'
checkpoint = ModelCheckpoint(filepath, monitor=['acc'], verbose=1, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(train_generator, epochs=5, workers=8, 
    steps_per_epoch=5000 // 64, shuffle=True, callbacks=callbacks_list)


img = preprocess_input(images[0].reshape(1, 96, 96, 3))

out = model.predict(img)

confidence = out[0]
class_prediction = np.argmax(list(out[0]))

print(confidence)
print(class_prediction)