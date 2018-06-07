from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Add, Input, Average
from keras.layers import Conv2D, Conv1D
from core.layers import NAIDE_Conv2D_Q1, NAIDE_Conv2D_Q2
from core.layers import NAIDE_Conv2D_DOWN1, NAIDE_Conv2D_DOWN2
from core.layers import NAIDE_Conv2D_E1, NAIDE_Conv2D_E2

from keras.optimizers import Adam

import tensorflow as tf
import numpy as np

import keras.backend as K

def fine_tuning_loss(y_true,y_pred):    #
    return K.mean(K.square(y_true[:,:,:,1]-(y_pred[:,:,:,0]*y_true[:,:,:,1]+y_pred[:,:,:,1])) + 2*y_pred[:,:,:,0]*K.square(y_true[:,:,:,2]) - K.square(y_true[:,:,:,2]))

units = 64
units_1by1 = 128

def make_model(img_x,img_y):
    
    input_shape = (img_x,img_y,1)

    #with tf.device("/cpu:0"):
        # initialize the model
    input_layer = Input(shape=input_shape)
        
    layer_A = input_layer
    layer_B = input_layer
    layer_C = input_layer

    layer_A = NAIDE_Conv2D_Q1(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = NAIDE_Conv2D_E1(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = NAIDE_Conv2D_DOWN1(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg1 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg1 = Activation('relu')(layer_avg1)
    layer_avg1_1by1 = layer_avg1
    layer_avg1_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg1_1by1)
    layer_avg1_1by1 = Activation('relu')(layer_avg1_1by1)
    layer_avg1_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg1_1by1)
    layer_avg1_1by1 = Average()([layer_avg1, layer_avg1_1by1])
    layer_avg1_1by1 = Activation('relu')(layer_avg1_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg2 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg2 = Activation('relu')(layer_avg2)
    layer_avg2_1by1 = layer_avg2
    layer_avg2_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg2_1by1)
    layer_avg2_1by1 = Activation('relu')(layer_avg2_1by1)
    layer_avg2_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg2_1by1)
    layer_avg2_1by1 = Average()([layer_avg2, layer_avg2_1by1])
    layer_avg2_1by1 = Activation('relu')(layer_avg2_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg3 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg3 = Activation('relu')(layer_avg3)
    layer_avg3_1by1 = layer_avg3
    layer_avg3_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg3_1by1)
    layer_avg3_1by1 = Activation('relu')(layer_avg3_1by1)
    layer_avg3_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg3_1by1)
    layer_avg3_1by1 = Average()([layer_avg3, layer_avg3_1by1])
    layer_avg3_1by1 = Activation('relu')(layer_avg3_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg4 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg4 = Activation('relu')(layer_avg4)
    layer_avg4_1by1 = layer_avg4
    layer_avg4_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg4_1by1)
    layer_avg4_1by1 = Activation('relu')(layer_avg4_1by1)
    layer_avg4_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg4_1by1)
    layer_avg4_1by1 = Average()([layer_avg4, layer_avg4_1by1])
    layer_avg4_1by1 = Activation('relu')(layer_avg4_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg5 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg5 = Activation('relu')(layer_avg5)
    layer_avg5_1by1 = layer_avg5
    layer_avg5_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg5_1by1)
    layer_avg5_1by1 = Activation('relu')(layer_avg5_1by1)
    layer_avg5_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg5_1by1)
    layer_avg5_1by1 = Average()([layer_avg5, layer_avg5_1by1])
    layer_avg5_1by1 = Activation('relu')(layer_avg5_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg6 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg6 = Activation('relu')(layer_avg6)
    layer_avg6_1by1 = layer_avg6
    layer_avg6_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg6_1by1)
    layer_avg6_1by1 = Activation('relu')(layer_avg6_1by1)
    layer_avg6_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg6_1by1)
    layer_avg6_1by1 = Average()([layer_avg6, layer_avg6_1by1])
    layer_avg6_1by1 = Activation('relu')(layer_avg6_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg7 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg7 = Activation('relu')(layer_avg7)
    layer_avg7_1by1 = layer_avg7
    layer_avg7_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg7_1by1)
    layer_avg7_1by1 = Activation('relu')(layer_avg7_1by1)
    layer_avg7_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg7_1by1)
    layer_avg7_1by1 = Average()([layer_avg7, layer_avg7_1by1])
    layer_avg7_1by1 = Activation('relu')(layer_avg7_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg8 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg8 = Activation('relu')(layer_avg8)
    layer_avg8_1by1 = layer_avg8
    layer_avg8_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg8_1by1)
    layer_avg8_1by1 = Activation('relu')(layer_avg8_1by1)
    layer_avg8_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg8_1by1)
    layer_avg8_1by1 = Average()([layer_avg8, layer_avg8_1by1])
    layer_avg8_1by1 = Activation('relu')(layer_avg8_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg9 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg9 = Activation('relu')(layer_avg9)
    layer_avg9_1by1 = layer_avg9
    layer_avg9_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg9_1by1)
    layer_avg9_1by1 = Activation('relu')(layer_avg9_1by1)
    layer_avg9_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg9_1by1)
    layer_avg9_1by1 = Average()([layer_avg9, layer_avg9_1by1])
    layer_avg9_1by1 = Activation('relu')(layer_avg9_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg10 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg10 = Activation('relu')(layer_avg10)
    layer_avg10_1by1 = layer_avg10
    layer_avg10_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg10_1by1)
    layer_avg10_1by1 = Activation('relu')(layer_avg10_1by1)
    layer_avg10_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg10_1by1)
    layer_avg10_1by1 = Average()([layer_avg10, layer_avg10_1by1])
    layer_avg10_1by1 = Activation('relu')(layer_avg10_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg11 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg11 = Activation('relu')(layer_avg11)
    layer_avg11_1by1 = layer_avg11
    layer_avg11_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg11_1by1)
    layer_avg11_1by1 = Activation('relu')(layer_avg11_1by1)
    layer_avg11_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg11_1by1)
    layer_avg11_1by1 = Average()([layer_avg11, layer_avg11_1by1])
    layer_avg11_1by1 = Activation('relu')(layer_avg11_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg12 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg12 = Activation('relu')(layer_avg12)
    layer_avg12_1by1 = layer_avg12
    layer_avg12_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg12_1by1)
    layer_avg12_1by1 = Activation('relu')(layer_avg12_1by1)
    layer_avg12_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg12_1by1)
    layer_avg12_1by1 = Average()([layer_avg12, layer_avg12_1by1])
    layer_avg12_1by1 = Activation('relu')(layer_avg12_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg13 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg13 = Activation('relu')(layer_avg13)
    layer_avg13_1by1 = layer_avg13
    layer_avg13_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg13_1by1)
    layer_avg13_1by1 = Activation('relu')(layer_avg13_1by1)
    layer_avg13_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg13_1by1)
    layer_avg13_1by1 = Average()([layer_avg13, layer_avg13_1by1])
    layer_avg13_1by1 = Activation('relu')(layer_avg13_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg14 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg14 = Activation('relu')(layer_avg14)
    layer_avg14_1by1 = layer_avg14
    layer_avg14_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg14_1by1)
    layer_avg14_1by1 = Activation('relu')(layer_avg14_1by1)
    layer_avg14_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg14_1by1)
    layer_avg14_1by1 = Average()([layer_avg14, layer_avg14_1by1])
    layer_avg14_1by1 = Activation('relu')(layer_avg14_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg15 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg15 = Activation('relu')(layer_avg15)
    layer_avg15_1by1 = layer_avg15
    layer_avg15_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg15_1by1)
    layer_avg15_1by1 = Activation('relu')(layer_avg15_1by1)
    layer_avg15_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg15_1by1)
    layer_avg15_1by1 = Average()([layer_avg15, layer_avg15_1by1])
    layer_avg15_1by1 = Activation('relu')(layer_avg15_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg16 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg16 = Activation('relu')(layer_avg16)
    layer_avg16_1by1 = layer_avg16
    layer_avg16_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg16_1by1)
    layer_avg16_1by1 = Activation('relu')(layer_avg16_1by1)
    layer_avg16_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg16_1by1)
    layer_avg16_1by1 = Average()([layer_avg16, layer_avg16_1by1])
    layer_avg16_1by1 = Activation('relu')(layer_avg16_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg17 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg17 = Activation('relu')(layer_avg17)
    layer_avg17_1by1 = layer_avg17
    layer_avg17_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg17_1by1)
    layer_avg17_1by1 = Activation('relu')(layer_avg17_1by1)
    layer_avg17_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg17_1by1)
    layer_avg17_1by1 = Average()([layer_avg17, layer_avg17_1by1])
    layer_avg17_1by1 = Activation('relu')(layer_avg17_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg18 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg18 = Activation('relu')(layer_avg18)
    layer_avg18_1by1 = layer_avg18
    layer_avg18_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg18_1by1)
    layer_avg18_1by1 = Activation('relu')(layer_avg18_1by1)
    layer_avg18_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg18_1by1)
    layer_avg18_1by1 = Average()([layer_avg18, layer_avg18_1by1])
    layer_avg18_1by1 = Activation('relu')(layer_avg18_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg19 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg19 = Activation('relu')(layer_avg19)
    layer_avg19_1by1 = layer_avg19
    layer_avg19_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg19_1by1)
    layer_avg19_1by1 = Activation('relu')(layer_avg19_1by1)
    layer_avg19_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg19_1by1)
    layer_avg19_1by1 = Average()([layer_avg19, layer_avg19_1by1])
    layer_avg19_1by1 = Activation('relu')(layer_avg19_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg20 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg20 = Activation('relu')(layer_avg20)
    layer_avg20_1by1 = layer_avg20
    layer_avg20_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg20_1by1)
    layer_avg20_1by1 = Activation('relu')(layer_avg20_1by1)
    layer_avg20_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg20_1by1)
    layer_avg20_1by1 = Average()([layer_avg20, layer_avg20_1by1])
    layer_avg20_1by1 = Activation('relu')(layer_avg20_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg21 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg21 = Activation('relu')(layer_avg21)
    layer_avg21_1by1 = layer_avg21
    layer_avg21_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg21_1by1)
    layer_avg21_1by1 = Activation('relu')(layer_avg21_1by1)
    layer_avg21_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg21_1by1)
    layer_avg21_1by1 = Average()([layer_avg21, layer_avg21_1by1])
    layer_avg21_1by1 = Activation('relu')(layer_avg21_1by1)

    layer_A = Activation('relu')(layer_A)
    layer_A = NAIDE_Conv2D_Q2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_A)

    layer_B = Activation('relu')(layer_B)
    layer_B = NAIDE_Conv2D_E2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_B)

    layer_C = Activation('relu')(layer_C)
    layer_C = NAIDE_Conv2D_DOWN2(units, (3, 3), kernel_initializer='he_uniform', padding='same')(layer_C)

    layer_A_avg = layer_A
    layer_B_avg = layer_B
    layer_C_avg = layer_C

    layer_avg22 = Average()([layer_A_avg, layer_B_avg, layer_C_avg])
    layer_avg22 = Activation('relu')(layer_avg22)
    layer_avg22_1by1 = layer_avg22
    layer_avg22_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg22_1by1)
    layer_avg22_1by1 = Activation('relu')(layer_avg22_1by1)
    layer_avg22_1by1 = Conv2D(units,(1,1), kernel_initializer='he_uniform',)(layer_avg22_1by1)
    layer_avg22_1by1 = Average()([layer_avg22, layer_avg22_1by1])
    layer_avg22_1by1 = Activation('relu')(layer_avg22_1by1)


    layer_ = Average()([layer_avg17_1by1, layer_avg16_1by1, layer_avg15_1by1, layer_avg14_1by1, 
                        layer_avg13_1by1, layer_avg12_1by1, layer_avg11_1by1, layer_avg10_1by1, 
                        layer_avg9_1by1, layer_avg8_1by1, layer_avg7_1by1, layer_avg6_1by1,
                        layer_avg5_1by1, layer_avg4_1by1, layer_avg3_1by1, layer_avg2_1by1,
                        layer_avg1_1by1, layer_avg18_1by1, layer_avg19_1by1, layer_avg20_1by1,
                        layer_avg21_1by1, layer_avg22_1by1])

    layer_ = Conv2D(units_1by1,(1,1), kernel_initializer='he_uniform',)(layer_)
    layer_ = Activation('relu')(layer_)

    layer_residual = layer_
    layer_reresidual_1 = layer_
    layer_residual = Conv2D(units_1by1,(1,1), kernel_initializer='he_uniform',)(layer_residual)
    layer_residual = Activation('relu')(layer_residual)
    layer_residual = Conv2D(units_1by1,(1,1), kernel_initializer='he_uniform',)(layer_residual)
    layer_ = Average()([layer_, layer_residual])
    layer_ = Activation('relu')(layer_)

    layer_residual = layer_
    layer_reresidual_2 = layer_
    layer_residual = Conv2D(units_1by1,(1,1), kernel_initializer='he_uniform',)(layer_residual)
    layer_residual = Activation('relu')(layer_residual)
    layer_residual = Conv2D(units_1by1,(1,1), kernel_initializer='he_uniform',)(layer_residual)
    layer_ = Average()([layer_, layer_residual])
    layer_ = Activation('relu')(layer_)

    layer_ = Average()([layer_, layer_reresidual_1, layer_reresidual_2])
    layer_ = Conv2D(2,(1,1), kernel_initializer='he_uniform',)(layer_)

    output_layer = layer_
    model = Model(inputs=[input_layer], outputs=[output_layer])
    
    adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    model.compile(loss=fine_tuning_loss, optimizer=adam)

    return model