from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
import tensorflow as tf


def generator_network(input_shape=(256, 256, 3)):
    """
    the image completion network architecture
    """

    model = Sequential()

    model.add(layers.Conv2D(64, kernel_size=5, strides=1, padding='same',
                            dilation_rate=(1, 1), input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding='same',
                            dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(128, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, kernel_size=3, strides=2, padding='same',
                            dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(4, 4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(8, 8)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(16, 16)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same',
                                     dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(128, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same',
                                     dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(32, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(3, kernel_size=3, strides=1, padding='same',
                            dilation_rate=(1, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('sigmoid'))

    return model


def discriminator_network(global_shape=(256, 256, 3), local_shape=(128, 128, 3)):

    def crop_image(img, crop):
        return tf.image.crop_to_bounding_box(img,
                                             crop[1],
                                             crop[0],
                                             crop[3] - crop[1],
                                             crop[2] - crop[0])

    in_pts = layers.Input(shape=(4,), dtype='int32') # [y1,x1,y2,x2]
    cropping = layers.Lambda(lambda x: K.map_fn(lambda y: crop_image(y[0], y[1]), elems=x, dtype=tf.float32),
                             output_shape=local_shape)

    global_img = layers.Input(shape=global_shape)
    local_img = cropping([global_img, in_pts])
    local_img.set_shape((None,) + local_shape)

    # global discriminator
    out_global = layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(global_img)
    out_global = layers.BatchNormalization()(out_global)
    out_global = layers.Activation('relu')(out_global)

    out_global = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(out_global)
    out_global = layers.BatchNormalization()(out_global)
    out_global = layers.Activation('relu')(out_global)

    out_global = layers.Conv2D(256, kernel_size=5, strides=2, padding='same')(out_global)
    out_global = layers.BatchNormalization()(out_global)
    out_global = layers.Activation('relu')(out_global)

    out_global = layers.Conv2D(512, kernel_size=5, strides=2, padding='same')(out_global)
    out_global = layers.BatchNormalization()(out_global)
    out_global = layers.Activation('relu')(out_global)

    out_global = layers.Conv2D(512, kernel_size=5, strides=2, padding='same')(out_global)
    out_global = layers.BatchNormalization()(out_global)
    out_global = layers.Activation('relu')(out_global)

    out_global = layers.Conv2D(512, kernel_size=5, strides=2, padding='same')(out_global)
    out_global = layers.BatchNormalization()(out_global)
    out_global = layers.Activation('relu')(out_global)

    out_global = layers.Flatten()(out_global)
    out_global = layers.Dense(1024, activation='relu')(out_global)

    # local discriminator
    out_local = layers.Conv2D(64, kernel_size=5, strides=2, padding='same')(local_img)
    out_local = layers.BatchNormalization()(out_local)
    out_local = layers.Activation('relu')(out_local)

    out_local = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(out_local)
    out_local = layers.BatchNormalization()(out_local)
    out_local = layers.Activation('relu')(out_local)

    out_local = layers.Conv2D(256, kernel_size=5, strides=2, padding='same')(out_local)
    out_local = layers.BatchNormalization()(out_local)
    out_local = layers.Activation('relu')(out_local)

    out_local = layers.Conv2D(512, kernel_size=5, strides=2, padding='same')(out_local)
    out_local = layers.BatchNormalization()(out_local)
    out_local = layers.Activation('relu')(out_local)

    out_local = layers.Conv2D(512, kernel_size=5, strides=2, padding='same')(out_local)
    out_local = layers.BatchNormalization()(out_local)
    out_local = layers.Activation('relu')(out_local)

    out_local = layers.Flatten()(out_local)
    out_local = layers.Dense(1024, activation='relu')(out_local)

    # concatenate local and global discriminator
    out = layers.concatenate([out_local, out_global])
    out = layers.Dense(1, activation='sigmoid')(out)

    return Model(inputs=[global_img, in_pts], outputs=out)


if __name__ == "__main__":
    generator = generator_network()
    generator.summary()
    discriminator = discriminator_network()
    discriminator.summary()
