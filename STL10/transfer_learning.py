import stl10_input as stl
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os

MODEL_NAME = 'stl_dn121_transfer'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
IMG_HEIGHT = 96
IMG_WIDTH = 96
IMG_CHANNELS = 3
BATCH_SIZE = 64
NUM_CLASSES = 10
CHECKPOINT_DIR = 'checkpoints/'
CLASS_NAMES_LOC = 'data/stl10_binary/class_names.txt'
DATA_PATH = 'data/stl10_binary/train_X.bin'
LABEL_PATH = 'data/stl10_binary/train_y.bin'
LAYER_UNITS = (128, 32)
LR = 1e-4
EPOCHS = 10
NUM_SAMPLES = 5000
NUM_VAL_SAMPLES = 256
NUM_TRAINING_SAMPLES = NUM_SAMPLES - NUM_VAL_SAMPLES

def build_model(base, layer_units, num_classes):
    for layer in base.layers:
        layer.trainable = False
    
    x = base.output
    x = Flatten()(x)
    for num_units in layer_units:
        x = Dense(num_units, activation='relu')(x)

    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=predictions)
    return model

# Download and Organize data
stl.download_and_extract()
images = stl.read_all_images(DATA_PATH)
labels = stl.read_labels(LABEL_PATH)

train_x = images[:NUM_TRAINING_SAMPLES]
train_y = labels[:NUM_TRAINING_SAMPLES]

val_x = images[-NUM_VAL_SAMPLES:]
val_y = labels[-NUM_VAL_SAMPLES:]

if not os.path.isdir(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)
if not os.path.isdir(VAL_DIR):
    os.makedirs(VAL_DIR)

stl.save_images(train_x, train_y, TRAIN_DIR)
stl.save_images(val_x, val_y, VAL_DIR)

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
train_gen = datagen.flow_from_directory(
    TRAIN_DIR, 
    target_size=(IMG_HEIGHT, IMG_WIDTH), 
    batch_size=BATCH_SIZE
)
val_gen = datagen.flow_from_directory(
    VAL_DIR, 
    target_size=(IMG_HEIGHT, IMG_WIDTH), 
    batch_size=BATCH_SIZE
)

class_names = dict()
with open(CLASS_NAMES_LOC, 'r') as f:
    for i in range(NUM_CLASSES):
        class_names[i] = f.readline().strip()

adam = Adam(lr=LR)
dn121 = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
model = build_model(dn121, LAYER_UNITS, NUM_CLASSES)
model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

if not os.path.isdir(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

checkpoint_path = CHECKPOINT_DIR + MODEL_NAME + '.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor=['acc'], verbose=1, mode='max')
callbacks = [checkpoint]

history = model.fit_generator(
    train_gen, 
    epochs=EPOCHS,
    steps_per_epoch=NUM_TRAINING_SAMPLES // BATCH_SIZE,
    validation_data=val_gen, 
    validation_steps=NUM_VAL_SAMPLES // BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks)



