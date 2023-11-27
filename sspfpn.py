import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Input, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
import numpy as np
import prepareVOC12 as voc
import matrix as mt
import pathlib



def customized_loss(input_img, y_true, y_pred, gamma=0.85):
    # calculate categorical crossentropy loss
    Lc = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    Lc_mean = tf.reduce_mean(Lc)

    # convert y_pred to RGB label maps
    y_pred_rgb = voc.batch_onehot_to_label(y_pred, voc.class_dict)

    # calculate cosine similarity loss for each image in the batch
    cosine_losses = []
    for i in range(input_img.shape[0]):
        single_image = input_img[i]
        single_y_pred = y_pred_rgb[i]
        cosine_loss = mt.calculate_cosine_similarity(single_image, single_y_pred)
        cosine_losses.append(cosine_loss)
    cosine_loss = tf.reduce_mean(cosine_losses)

    # combine losses using gamma
    cosine_loss = tf.cast(cosine_loss, tf.float32)
    loss = gamma * Lc_mean + (1 - gamma) * cosine_loss

    return loss
 

def SSPFPN(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)

    # Freeze VGG19 layers
    for layer in vgg19.layers:
        layer.trainable = False

    """ feature maps """
    c1 = vgg19.get_layer("block1_conv2").output         ## (224 x 224)
    c2 = vgg19.get_layer("block2_conv2").output         ## (112 x 112)
    c3 = vgg19.get_layer("block3_conv4").output         ## (56 x 56)
    c4 = vgg19.get_layer("block4_conv4").output         ## (28 x 28)

    """ Bridge """
    c5 = vgg19.get_layer("block5_conv4").output         ## (14 x 14)

    """ SSPFPN """
    P5 = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', activation='relu')(c5)
    P5_upsampled = tf.keras.layers.UpSampling2D()(P5)

    P4 = tf.keras.layers.Concatenate()([c4, P5_upsampled])
    P4 = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', activation='relu')(P4)
    P4_upsampled = tf.keras.layers.UpSampling2D()(P4)

    P3 = tf.keras.layers.Concatenate()([c3, P4_upsampled])
    P3 = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', activation='relu')(P3)
    P3_upsampled = tf.keras.layers.UpSampling2D()(P3)

    P2 = tf.keras.layers.Concatenate()([c2, P3_upsampled])
    P2 = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', activation='relu')(P2)
    P2_upsampled = tf.keras.layers.UpSampling2D()(P2)

    P1 = tf.keras.layers.Concatenate()([c1, P2_upsampled])
    P1 = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', activation='relu')(P1)
    
    """ Output """
    outputs = Conv2D(21, 1, padding="same", activation="sigmoid")(P1)

    # """ Optimizier Tunning in the Papar """
    # opt = SGD(learning_rate=2.5e-4, momentum=0.9, decay=5e-4)

    """ Set up Model """

    model = Model(inputs, outputs)
    
    return model


