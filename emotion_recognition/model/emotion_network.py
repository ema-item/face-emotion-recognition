from tensorflow.keras import layers, models, regularizers, initializers

kernel_init = initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')
bias_init = initializers.Zeros()

l1_l2_reg = regularizers.l1_l2(l1=0.0, l2=0.01)  


def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = models.Sequential(name="sequential_1")

    # Conv block 1
    model.add(layers.Conv2D(
        filters=64, kernel_size=(3,3), strides=(1,1), padding='valid',
        activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init,
        kernel_regularizer=l1_l2_reg,
        name='conv2d_1',
        input_shape=input_shape, dtype='float32'
    ))
    model.add(layers.Conv2D(
        filters=64, kernel_size=(3,3), strides=(1,1), padding='same',
        activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init,
        name='conv2d_2'
    ))
    model.add(layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, name='batch_normalization_1',
        gamma_initializer=initializers.Ones(),
        beta_initializer=initializers.Zeros(),
        moving_mean_initializer=initializers.Zeros(),
        moving_variance_initializer=initializers.Ones()
    ))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max_pooling2d_1'))
    model.add(layers.Dropout(0.5, name='dropout_1'))

    # Conv block 2
    model.add(layers.Conv2D(
        filters=128, kernel_size=(3,3), padding='same',
        activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init,
        name='conv2d_3'
    ))
    model.add(layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, name='batch_normalization_2',
        gamma_initializer=initializers.Ones(),
        beta_initializer=initializers.Zeros(),
        moving_mean_initializer=initializers.Zeros(),
        moving_variance_initializer=initializers.Ones()
    ))
    model.add(layers.Conv2D(
        filters=128, kernel_size=(3,3), padding='same',
        activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init,
        name='conv2d_4'
    ))
    model.add(layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, name='batch_normalization_3',
        gamma_initializer=initializers.Ones(),
        beta_initializer=initializers.Zeros(),
        moving_mean_initializer=initializers.Zeros(),
        moving_variance_initializer=initializers.Ones()
    ))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max_pooling2d_2'))
    model.add(layers.Dropout(0.5, name='dropout_2'))

    # Conv block 3
    model.add(layers.Conv2D(
        filters=256, kernel_size=(3,3), padding='same',
        activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init,
        name='conv2d_5'
    ))
    model.add(layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, name='batch_normalization_4',
        gamma_initializer=initializers.Ones(),
        beta_initializer=initializers.Zeros(),
        moving_mean_initializer=initializers.Zeros(),
        moving_variance_initializer=initializers.Ones()
    ))
    model.add(layers.Conv2D(
        filters=256, kernel_size=(3,3), padding='same',
        activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init,
        name='conv2d_6'
    ))
    model.add(layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, name='batch_normalization_5',
        gamma_initializer=initializers.Ones(),
        beta_initializer=initializers.Zeros(),
        moving_mean_initializer=initializers.Zeros(),
        moving_variance_initializer=initializers.Ones()
    ))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max_pooling2d_3'))
    model.add(layers.Dropout(0.5, name='dropout_3'))

    # Conv block 4
    model.add(layers.Conv2D(
        filters=512, kernel_size=(3,3), padding='same',
        activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init,
        name='conv2d_7'
    ))
    model.add(layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, name='batch_normalization_6',
        gamma_initializer=initializers.Ones(),
        beta_initializer=initializers.Zeros(),
        moving_mean_initializer=initializers.Zeros(),
        moving_variance_initializer=initializers.Ones()
    ))
    model.add(layers.Conv2D(
        filters=512, kernel_size=(3,3), padding='same',
        activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init,
        name='conv2d_8'
    ))
    model.add(layers.BatchNormalization(
        axis=-1, momentum=0.99, epsilon=0.001, name='batch_normalization_7',
        gamma_initializer=initializers.Ones(),
        beta_initializer=initializers.Zeros(),
        moving_mean_initializer=initializers.Zeros(),
        moving_variance_initializer=initializers.Ones()
    ))
    model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='max_pooling2d_4'))
    model.add(layers.Dropout(0.5, name='dropout_4'))

    # Classifier
    model.add(layers.Flatten(name='flatten_1'))
    model.add(layers.Dense(512, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, name='dense_1'))
    model.add(layers.Dropout(0.4, name='dropout_5'))
    model.add(layers.Dense(256, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, name='dense_2'))
    model.add(layers.Dropout(0.4, name='dropout_6'))
    model.add(layers.Dense(128, activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, name='dense_3'))
    model.add(layers.Dropout(0.5, name='dropout_7'))
    model.add(layers.Dense(num_classes, activation='softmax', kernel_initializer=kernel_init, bias_initializer=bias_init, name='dense_4'))

    return model
# fer2013 weight
weights_path = "D:/privet/face_detection_model/emotion_recognition/model/fer2013_weight.h5"
# model
model = build_model()
# load weight in model
model.load_weights(weights_path)
# save model
model.save('emotion_model.h5')