# Malaria Detection using TensorFlow

This repository contains code for detecting malaria using TensorFlow. The code is implemented in a Jupyter notebook (`malaria_detection_tf.ipynb`).

## Dataset

The malaria dataset used in this project is obtained from TensorFlow Datasets (`tfds`). It consists of images categorized into two classes: Parasitized and Uninfected.

## Model Architecture

The convolutional neural network (CNN) model architecture used for malaria detection consists of several layers:

- Input Layer: Accepts images of size `(224, 224, 3)`
- Convolutional Layers: Two sets of convolutional layers with batch normalization and max-pooling.
- Flatten Layer: Flattens the output from convolutional layers.
- Dense Layers: Two dense layers with batch normalization.
- Output Layer: Single neuron with sigmoid activation function for binary classification.

## Data Preprocessing

- The images are resized to `(224, 224)` and normalized to a range of `[0, 1]`.
- Data is split into training, validation, and test sets with ratios of 80%, 10%, and 10% respectively.

## Training

The model is compiled using the Adam optimizer with a learning rate of 0.01 and binary cross-entropy loss function. Training is conducted for 25 epochs with both training and validation accuracy monitored.

## Evaluation

The trained model is evaluated on the test dataset to measure its performance. Evaluation metrics include loss and accuracy.

## Prediction

The model is used to predict whether a given image contains malaria parasites or not. Predictions are made on sample images from the test dataset, and results are visualized.

## Usage

To run the code:

1. Install the required dependencies mentioned in the notebook.
2. Execute the notebook (`malaria_detection_tf.ipynb`) in a compatible environment.

## References

- Original notebook: [malaria_detection_tf.ipynb](https://colab.research.google.com/drive/18duKHf0dmEt5JuZJiLVGb6k_9v44rSXj)
- TensorFlow Datasets: [tfds](https://www.tensorflow.org/datasets)

Feel free to contribute and improve upon this codebase!
