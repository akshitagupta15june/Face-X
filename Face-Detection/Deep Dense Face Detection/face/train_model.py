"""
Script for training models
"""

import os

import keras

import face.utilities
import face.models
import face.data_generators
import face.config


def get_callbacks():

    model_path = face.config.model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, verbose=1)

    reduce_learning_rate_callback = keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=2, verbose=1)
    early_stop_callback = keras.callbacks.EarlyStopping(patience=8, verbose=1)

    return [model_checkpoint, reduce_learning_rate_callback, early_stop_callback]


def main():

    # dataset = "large_dataset"
    dataset = "medium_dataset"
    # dataset = "small_dataset"

    data_directory = os.path.join(face.config.data_directory, dataset)

    training_image_paths_file = os.path.join(data_directory, "training_image_paths.txt")
    training_bounding_boxes_file = os.path.join(data_directory, "training_bounding_boxes_list.txt")

    validation_image_paths_file = os.path.join(data_directory, "validation_image_paths.txt")
    validation_bounding_boxes_file = os.path.join(data_directory, "validation_bounding_boxes_list.txt")

    batch_size = face.config.batch_size

    model = face.models.get_pretrained_vgg_model(image_shape=face.config.image_shape)
    # model.load_weights(face.config.model_path)

    training_data_generator = face.data_generators.get_batches_generator(
        training_image_paths_file, training_bounding_boxes_file, batch_size, face.config.crop_size)

    validation_data_generator = face.data_generators.get_batches_generator(
        validation_image_paths_file, validation_bounding_boxes_file, batch_size, face.config.crop_size)

    model.fit_generator(
        training_data_generator, samples_per_epoch=face.utilities.get_file_lines_count(training_image_paths_file),
        nb_epoch=100,
        validation_data=validation_data_generator,
        nb_val_samples=face.utilities.get_file_lines_count(validation_image_paths_file),
        callbacks=get_callbacks()
    )


if __name__ == "__main__":

    main()
