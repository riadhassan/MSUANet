import sys, os
import numpy as np
# from matplotlib import pyplot as plt
# from scipy.io import loadmat
# from skimage import io
from keras_preprocessing.image import transform_matrix_offset_center, Iterator, random_channel_shift, flip_axis
from keras.preprocessing.image import ImageDataGenerator

def augment_image_label(x, y, imsize=256, trans_threshold=0.0, horizontal_flip=None, rotation_range=None,
                        height_shift_range=None, width_shift_range=None, shear_range=None, zoom_range=None,
                        elastic=None, add_noise=None):  # 2D image

    x = np.reshape(x, (1, imsize, imsize))
    y = np.reshape(y, (1, imsize, imsize))  # force to reshape
    h = imsize
    w = imsize
    row_index = 1
    col_index = 2

    if horizontal_flip is not None:
        if np.random.random() < trans_threshold:
            x = flip_axis(x, 2)
            y = flip_axis(y, 2)
    tep2 = np.random.random()
    if tep2 < trans_threshold:

        if rotation_range is not None:
            theta = np.pi / 180.0 * np.random.uniform(-rotation_range, rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if height_shift_range is not None:
            tx = np.random.uniform(-height_shift_range, height_shift_range) * x.shape[row_index]
        else:
            tx = 0

        if width_shift_range is not None:
            ty = np.random.uniform(-width_shift_range, width_shift_range) * x.shape[col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if shear_range is not None:
            shear = np.random.uniform(-shear_range, shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        transform_parameters = {'theta': theta, 'tx': height_shift_range,
                                'ty': width_shift_range, 'zx': zoom_range[0],
                                'zy': zoom_range[1], 'flip_horizontal': horizontal_flip}

        img_gen = ImageDataGenerator()
        x = img_gen.apply_transform(x, transform_parameters)  # ,fill_mode='nearest')

        y = img_gen.apply_transform(y, transform_parameters)  # ,# fill_mode='nearest')


    tep3 = np.random.random()
    if add_noise is not None:
        if tep3 < trans_threshold:
            x = x + 0.15 * x.std() * np.random.random(x.shape)
    # return x.reshape(1,1,imsize,imsize), y.reshape(1,1,imsize,imsize)
    return x.reshape(1, imsize, imsize), y.reshape(1, imsize, imsize)