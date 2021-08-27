'''
This file contains defintions for the used models.
The models proposed by Zaid et al. all start with 'zaid_' and were taken from:
https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA

The simplified models in which the first convolutional layer is removed start with 'noConv1_'.
'''

from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from keras import backend as K
from keras.optimizers import Adam, RMSprop


#### MLP Best model (6 layers of 200 units)
def ascad_mlp_best(input_size=700, learning_rate=0.00001, classes=256):
    assert learning_rate == 0.00001, "Do not change learning rate ... keep it default"
    layer_nb = 6
    node = 200
    model = Sequential()
    model.add(Dense(node, input_dim=input_size, activation='relu'))
    for i in range(layer_nb-2):
        model.add(Dense(node, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


### CNN Best model
def ascad_cnn_best(input_size=700, learning_rate=0.00001, classes=256):
    assert learning_rate == 0.00001, "Do not change learning rate ... keep it default"
    # From VGG16 design
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # Block 2
    x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # Block 3
    x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # Block 4
    x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # Block 5
    x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='cnn_best')
    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def zaid_ascad_desync_0(input_size=700,learning_rate=0.00001,classes=256):
	# Designing input layer
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        
    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
        
    # Logits layer              
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='ascad')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def zaid_ascad_desync_50(input_size=700,learning_rate=0.00001,classes=256):
    # Personal design
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(32, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    # 2nd convolutional block
    x = Conv1D(64, 25, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(25, strides=25, name='block2_pool')(x)

    # 3rd convolutional block
    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)      
    x = BatchNormalization()(x)
    x = AveragePooling1D(4, strides=4, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification part
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(15, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='cnn_best')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def zaid_ascad_desync_100(input_size=700,learning_rate=0.00001,classes=256):
    # Personal design
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(32, 1, kernel_initializer='he_uniform', activation='linear', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    # 2nd convolutional block
    x = Conv1D(64, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(50, strides=50, name='block2_pool')(x)

    # 3rd convolutional block
    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)      
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification part
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='cnn_best')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def zaid_dpav4(input_size=4000,learning_rate=0.00001,classes=256):
    # Designing input layer
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='dpacontest_v4')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def zaid_aes_rd(input_size=700,learning_rate=0.00001,classes=256):
    # Designing input layer
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(8, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    # 2nd convolutional block
    x = Conv1D(16, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(50, strides=50, name='block2_pool')(x)

    # 3rd convolutional block
    x = Conv1D(32, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(7, strides=7, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)

    # Logits layer      
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='aes_rd')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def zaid_aes_hd(input_size=1250,learning_rate=0.00001,classes=256):
    # Designing input layer
    input_shape = (input_size,1)
    img_input = Input(shape=input_shape)

    # 1st convolutional block
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_norm1')(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='aes_hd_model')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_ascad_desync_0(input_size=700,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)
    x = Flatten(name='flatten')(x)

    x = Dense(10, activation='selu', name='fc1')(x)
    x = Dense(10, activation='selu', name='fc2')(x)          
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_ascad_desync_0')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_ascad_desync_50(input_size=700,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)

    x = Conv1D(64, 25, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(25, strides=25, name='block1_pool')(x)

    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv')(x)      
    x = BatchNormalization()(x)
    x = AveragePooling1D(4, strides=4, name='block2_pool')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_ascad_desync_50')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_ascad_desync_100(input_size=700,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)

    x = Conv1D(64, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(50, strides=50, name='block2_pool')(x)

    x = Conv1D(128, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv')(x)      
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(20, kernel_initializer='he_uniform', activation='selu', name='fc3')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_ascad_desync_100')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_dpav4(input_size=4000,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)

    x = Flatten(name='flatten')(x)

    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_dpav4')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_aes_rd(input_size=700,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)

    x = Conv1D(16, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(50, strides=50, name='block1_pool')(x)

    x = Conv1D(32, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv')(x)
    x = BatchNormalization()(x)
    x = AveragePooling1D(7, strides=7, name='block2_pool')(x)

    x = Flatten(name='flatten')(x)

    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_aes_rd')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def noConv1_aes_hd(input_size=1250,learning_rate=0.00001,classes=256):
    trace_input = Input(shape=(input_size,1))
    x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)

    x = Flatten(name='flatten')(x)

    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(trace_input, x, name='noConv1_aes_hd')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    return model


def aisy_ascad_f_hw_mlp(input_size=700, learning_rate=1e-5, classes=9):
    assert classes == 9
    img_input = Input(shape=(input_size, 1))
    x = Flatten(name='flatten')(img_input)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(760, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(704, activation='relu')(x)
    x = Dense(1016, activation='relu')(x)
    x = Dense(560, activation='relu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model


def aisy_ascad_f_id_mlp(input_size=700, learning_rate=1e-5, classes=256):
    assert classes == 256
    img_input = Input(shape=(input_size, 1))
    x = Flatten(name='flatten')(img_input)
    x = Dense(480, activation='elu')(x)
    x = Dense(480, activation='elu')(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(img_input, x)
    optimizer = RMSprop(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model