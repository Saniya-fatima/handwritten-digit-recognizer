
# Handwritten Digit Recognition

This project focuses on recognizing handwritten digits using a Convolutional Neural Network (CNN) implemented with TensorFlow and Keras.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [License](#license)

## Introduction
Handwritten digit recognition is a classic problem in the field of machine learning and computer vision. This project aims to build a CNN model to accurately classify digits from 0 to 9. The model is trained and tested on the MNIST dataset.

## Dataset
The dataset used is the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which contains 60,000 training images and 10,000 testing images of handwritten digits. Each image is a 28x28 grayscale image.

## Project Structure
The project repository contains the following files:

- `handwritten_digit_recognizer.ipynb`: Jupyter notebook with the complete analysis and model implementation.
- `README.md`: This readme file.

## Installation
To run this project locally, you need to have Python and Jupyter Notebook installed. You can install the required libraries using pip:

```bash
pip install numpy matplotlib tensorflow keras
```

## Data Preprocessing
Data preprocessing steps include:

- Normalizing the pixel values of the images to the range [0, 1].
- Reshaping the images to add a channel dimension for the CNN.
- One-hot encoding the labels.

## Model Architecture
The CNN model architecture includes:

- Convolutional layers for feature extraction.
- MaxPooling layers for downsampling.
- Dense layers for classification.
- Dropout layers to prevent overfitting.

The model summary is as follows:

```
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2D) (None, 5, 5, 64)         0
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0
_________________________________________________________________
dense (Dense)                (None, 128)               204928
_________________________________________________________________
dropout (Dropout)            (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 224,034
Trainable params: 224,034
Non-trainable params: 0
_________________________________________________________________
```

## Training
The model is trained using the following parameters:

- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy
- Number of epochs: 10
- Batch size: 128

## Evaluation
The model's performance is evaluated using the accuracy metric on the test set. The following evaluation steps are included:

- Plotting the training and validation accuracy/loss.
- Displaying the confusion matrix for the test set.
- Visualizing some of the test predictions.

## Conclusion
The CNN model achieves high accuracy on the MNIST dataset, demonstrating its effectiveness in recognizing handwritten digits. Further improvements could include experimenting with different architectures, data augmentation, and hyperparameter tuning.

