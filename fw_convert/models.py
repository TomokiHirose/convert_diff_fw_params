import torch
import tensorflow as tf
from logzero import logger
from torchvision import models
from torchsummary import summary
import torch.nn.functional as F
import random
import numpy as np

torch.backends.cudnn.enabled = False


def tf_vgg_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="same")(x)
    x = tf.keras.layers.Flatten()(tf.keras.layers.Permute((3, 1, 2))(x))
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    predictions = tf.keras.layers.Dense(1000, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    # model.summary()
    return model


def torch_vgg_model():
    model = models.vgg11(pretrained=True)
    #summary(model, (3, 224, 224))
    return model


class TorchSimpleFCNet(torch.nn.Module):
    def __init__(self):
        super(TorchSimpleFCNet, self).__init__()
        self.fc1 = torch.nn.Linear(784, 1000)
        self.fc2 = torch.nn.Linear(1000, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def TFSimpleFCNet():
    inputs = tf.keras.Input(shape=(784))
    x = tf.keras.layers.Dense(1000, activation="relu")(inputs)
    predictions = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=predictions)


class TorchConvFlattenNet(torch.nn.Module):
    def __init__(self):
        super(TorchConvFlattenNet, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)


def TFConvFlattenNet():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Flatten()(inputs)
    return tf.keras.Model(inputs=inputs, outputs=x)


class TorchFlattenNet(torch.nn.Module):
    def __init__(self):
        super(TorchFlattenNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        return torch.flatten(x, 1)


def TFFlattenNet():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3)(inputs)
    x = tf.keras.layers.Flatten()(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


class TorchSimpleNet(torch.nn.Module):
    def __init__(self):
        super(TorchSimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def TFSimpleNet():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    predictions = tf.keras.layers.Dense(10, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=predictions)


class TorchSimplePermuteNet(torch.nn.Module):
    def __init__(self):
        super(TorchSimplePermuteNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def TFSimplePermuteNet():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Flatten()(tf.keras.layers.Permute((3, 1, 2))(x))
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    predictions = tf.keras.layers.Dense(10, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=predictions)


"""
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


model = SimpleNet()
summary(model, (1, 28, 28))

inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inputs)
x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
x = tf.keras.layers.MaxPool2D(2)(x)
x = tf.keras.layers.Flatten()(tf.keras.layers.Permute((3,1,2))(x))
x = tf.keras.layers.Dense(128, activation="relu")(x)
predictions = tf.keras.layers.Dense(10, activation="softmax")(x)

tfmodel = tf.keras.Model(inputs=inputs, outputs=predictions)
tfmodel.summary()
"""