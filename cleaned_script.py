import os
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
import random
import time

import gdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import plotly.express as px

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Activation, MaxPooling2D, Dropout, Flatten, Dense, Conv2D, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, DenseNet121
from tensorflow.keras import optimizers

from scikeras.wrappers import KerasClassifier
from imgaug import augmenters as iaa

# (optional) pyngrok for launching streamlit if you use that part
from pyngrok import ngrok

# -------------------------
# Augmentation helpers (imgaug)
# -------------------------
def augment(data, augmenter):
    """augmenter is an imgaug augmenter; data either HWC or NHWC"""
    if len(data.shape) == 3:  # single image HWC
        augmented = augmenter(image=data)
        return augmented
    elif len(data.shape) == 4:  # batch NHWC
        augmented = augmenter(images=data)
        return augmented
    else:
        raise ValueError("Data must be 3D (H,W,C) or 4D (N,H,W,C).")

def rotate(data, angle):
    fun = iaa.Affine(rotate=angle)
    return augment(data, fun)

def shear(data, shear_val):
    fun = iaa.Affine(shear=shear_val)
    return augment(data, fun)

def scale(data, scale_val):
    fun = iaa.Affine(scale=scale_val)
    return augment(data, fun)

def flip_left_right(data, prob):
    fun = iaa.Fliplr(p=prob)
    return augment(data, fun)

def flip_up_down(data, prob):
    fun = iaa.Flipud(p=prob)
    return augment(data, fun)

def remove_color(data, channel):
    new_data = data.copy()
    if len(data.shape) == 3:
        new_data[:, :, channel] = 0
    elif len(data.shape) == 4:
        new_data[..., channel] = 0
    return new_data

# -------------------------
# Data utilities
# -------------------------
def get_metadata(metadata_path, which_splits=('train', 'test')):
    """Return metadata filtered to chosen splits (expects CSV with 'split','index','class')."""
    metadata = pd.read_csv(metadata_path)
    mask = metadata['split'].isin(which_splits)
    return metadata[mask].reset_index(drop=True)

def get_data_split(split_name, flatten, all_data, metadata, image_shape):
    """
    Get images and labels for a split.
    - all_data: numpy array with shape (N, H, W, C)
    - metadata: DataFrame with columns 'index' and 'class' and 'split'
    """
    sub_df = metadata[metadata['split'] == split_name].reset_index(drop=True)
    indices = sub_df['index'].values.astype(int)
    labels = sub_df['class'].values
    data = all_data[indices]
    if flatten:
        data = data.reshape((data.shape[0], -1))
    return data, labels

def get_train_data(flatten, all_data, metadata, image_shape):
    return get_data_split('train', flatten, all_data, metadata, image_shape)

def get_test_data(flatten, all_data, metadata, image_shape):
    return get_data_split('test', flatten, all_data, metadata, image_shape)

def get_field_data(flatten, all_data, metadata, image_shape):
    field_data, field_labels = get_data_split('field', flatten, all_data, metadata, image_shape)
    # replicate channel 0 into other channels if grayscale-ish
    field_data = field_data.copy()
    field_data[..., 2] = field_data[..., 0]
    field_data[..., 1] = field_data[..., 0]

    # make the field data messier using random augmentation per image
    for i in range(len(field_data)):
        rand = random.uniform(-1, 1)
        image = field_data[i]
        if abs(rand) < 0.5:
            image = rotate(image, angle=rand * 40)
        elif abs(rand) < 0.8:
            image = shear(image, shear_val=rand * 40)
        else:
            image = flip_left_right(image, prob=0.5)
        field_data[i] = image
    return field_data, field_labels

# -------------------------
# Plotting & helpers
# -------------------------
def plot_one_image(data, labels=None, index=None, image_shape=(64, 64, 3)):
    """
    Display a single image.
    - data can be (H,W,C) or (N,H,W,C)
    - labels: single label or list/array
    - index: for batch data, choose which image to show
    """
    if labels is None:
        labels = []
    num_dims = len(data.shape)

    if num_dims == 1:
        # flattened vector
        data = data.reshape(image_shape)
        num_dims = 3

    if num_dims == 3:
        image = data
        label = labels if (isinstance(labels, (str, int)) or len(labels) == 1) else ''
    elif num_dims == 4:
        if index is None:
            index = 0
        image = data[index]
        label = labels[index] if len(labels) > index else ''
    else:
        raise ValueError("Unsupported data shape for plotting.")

    plt.figure(figsize=(4, 4))
    plt.title(f"Label: {label}")
    plt.axis('off')
    # Convert image data type for plotting if necessary (assuming data might be float from normalization)
    plot_img = image.copy()
    if plot_img.max() <= 1.0:
        plot_img = (plot_img * 255).astype(np.uint8)
        
    plt.imshow(plot_img)
    plt.show()

def combine_data(data_list, labels_list):
    return np.concatenate(data_list, axis=0), np.concatenate(labels_list, axis=0)

def plot_acc(history, xlabel='Epoch #'):
    """history: Keras History object"""
    hist_dict = history.history
    epochs = range(1, len(hist_dict.get('loss', [])) + 1)
    df = pd.DataFrame({
        'epoch': list(epochs),
        'accuracy': hist_dict.get('accuracy', []),
        'val_accuracy': hist_dict.get('val_accuracy', [])
    })
    if df.empty:
        print("No history to plot.")
        return
    best_epoch = df.loc[df['val_accuracy'].idxmax(), 'epoch']
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.lineplot(x='epoch', y='val_accuracy', data=df, label='Validation', ax=ax)
    sns.lineplot(x='epoch', y='accuracy', data=df, label='Training', ax=ax)
    ax.axhline(0.5, linestyle='--', color='red', label='Chance')
    ax.axvline(x=best_epoch, linestyle='--', color='green', label=f'Best epoch ({int(best_epoch)})')
    ax.legend(loc=4)
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (Fraction)')
    plt.show()

# -------------------------
# Models
# -------------------------
def DenseClassifier(hidden_layer_sizes, nn_params):
    model = Sequential()
    model.add(Flatten(input_shape=nn_params['input_shape']))
    model.add(Dropout(0.5))
    for units in hidden_layer_sizes:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.5))
    model.add(Dense(units=nn_params['output_neurons'], activation=nn_params['output_activation']))

    model.compile(
        loss=nn_params['loss'],
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.95),
        metrics=['accuracy']
    )
    return model

def CNNClassifier(num_hidden_layers, nn_params):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=nn_params['input_shape'], padding='same',
                     kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(max(0, num_hidden_layers - 1)):
        model.add(Conv2D(64, (3, 3), padding='same',
                         kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(units=nn_params['output_neurons'], activation=nn_params['output_activation']))

    opt = tf.keras.optimizers.RMSprop(learning_rate=1e-5)
    model.compile(loss=nn_params['loss'], optimizer=opt, metrics=['accuracy'])
    return model

def TransferClassifier(name, nn_params, trainable=False):
    expert_dict = {
        'VGG16': VGG16,
        'VGG19': VGG19,
        'ResNet50': ResNet50,
        'DenseNet121': DenseNet121
    }
    if name not in expert_dict:
        raise ValueError(f"Unknown transfer model: {name}")
    expert_conv = expert_dict[name](weights='imagenet', include_top=False, input_shape=nn_params['input_shape'])
    for layer in expert_conv.layers:
        layer.trainable = trainable

    model = Sequential([
        expert_conv,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(nn_params['output_neurons'], activation=nn_params['output_activation'])
    ])

    model.compile(
        loss=nn_params['loss'],
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
        metrics=['accuracy']
    )
    return model

# -------------------------
# Example / wiring & usage (Main Training Block)
# -------------------------
if __name__ == "__main__":
    # --- Project Variables ---
    metadata_url = ("https://storage.googleapis.com/inspirit-ai-data-bucket-1/"
                    "Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/"
                    "Project%20-%20(Healthcare%20A)%20Pneumonia/metadata.csv")
    image_data_url = ("https://storage.googleapis.com/inspirit-ai-data-bucket-1/"
                      "Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/"
                      "Project%20-%20(Healthcare%20A)%20Pneumonia/image_data.npy")
    image_data_path = './image_data.npy'
    metadata_path = './metadata.csv'
    image_shape = (64, 64, 3)

    nn_params = {
        'input_shape': image_shape,
        'output_neurons': 1,
        'loss': 'binary_crossentropy',
        'output_activation': 'sigmoid'
    }

    # --- Data Download and Load ---
    if not os.path.exists(metadata_path):
        print("Downloading metadata...")
        os.system(f"wget -q -O {metadata_path} '{metadata_url}'")
    if not os.path.exists(image_data_path):
        print("Downloading image data...")
        os.system(f"wget -q -O {image_data_path} '{image_data_url}'")

    _all_data = np.load(image_data_path)
    _metadata = get_metadata(metadata_path, ['train', 'test']) 

    # --- Load Full Training and Test Data ---
    X_train_full, y_train_full = get_data_split('train', flatten=False, all_data = _all_data, metadata = _metadata, image_shape = image_shape)
    y_train_full = np.array([int(x) for x in y_train_full])
    
    X_test_full, y_test_full = get_data_split('test', flatten=False, all_data = _all_data, metadata = _metadata, image_shape = image_shape)
    y_test_full = np.array([int(x) for x in y_test_full])
    
    # --- Model Definition (Transfer Learning VGG16) ---
    def build_transfer_model():
        return TransferClassifier('VGG16', nn_params)

    # --- Training Setup ---
    # ModelCheckpoint will save the best model weights to model.h5
    checkpoint = ModelCheckpoint(
        'model.h5', 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1,
        mode='max'
    )
    
    # Use KerasClassifier for Scikit-learn interface
    clf = KerasClassifier(
        model=build_transfer_model,
        epochs=15, # Set a reasonable number of epochs
        batch_size=32,
        callbacks=[checkpoint],
        verbose=1,
        validation_split=0.1 # Use 10% of training data for validation
    )

    # --- Train and Save ---
    print("\nStarting full Transfer Learning model fit...")
    history = clf.fit(X_train_full, y_train_full)
    
    # Plotting is done on the full history object
    plot_acc(history)

    # --- Final Evaluation ---
    # We evaluate the performance of the model from the last epoch
    final_score = clf.score(X_test_full, y_test_full)
    print(f"\nFinal Test Accuracy (last epoch): {final_score:.4f}")
    
    print("\nTraining complete. The best model is saved as model.h5, ready for deployment.")
