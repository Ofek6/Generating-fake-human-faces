#importing all of the necessarily libaries
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import re
from keras_preprocessing.image import img_to_array
from google.colab import drive
import time


# the function is used to get the files in proper order(Sorting data alphabetically and numerically)
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


# defining the size of the image
SIZE = 128
_img = []
#the path is leading to the images that in my google drive
path = '/content/drive/MyDrive/Test'
files = os.listdir(path)
files = sorted_alphanumeric(files)
#this loop is looping through images in a directory, resizizing and converting them to a normalized array format for use in training a GAN"
for i in tqdm(files):
    if i == 'seed9090.png':
        break
    else:
        img = cv2.imread(path + '/' + i, 1)
        # open cv reads images in BGR format so we have to convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resizing image
        img = cv2.resize(img, (SIZE, SIZE))
        img = (img - 127.5) / 127.5
        img = img.astype(float)
        _img.append(img_to_array(img))

#the function is used to Plot a Grid of Images
def plot_images(sqr):
    plt.figure(figsize=(10, 10))
    plt.title("Real Images", fontsize=35)
    for i in range(sqr * sqr):
        plt.subplot(sqr, sqr, i + 1)
        plt.imshow(_img[i] * 0.5 + 0.5)
        plt.xticks([])
        plt.yticks([])

# to plot images
plot_images(6)
plt.show()

#defining the batch size and the dataset
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(np.array(_img)).batch(batch_size)

latent_dim = 100

"""This function defines a model for the generator network in a Generative Adversarial Network (GAN).
The model consists of a series of layers including Conv2DTranspose layers, Conv2D layers, BatchNormalization layers, and LeakyReLU activation layers.
The model is designed to take in a latent vector of noise and upsample it to generate an image.
The model uses a combination of convolutional and transposed convolutional layers to upsample the input noise, 
with batch normalization and leaky ReLU activation layers to stabilize the training process and improve the performance of the network.
When called, this function will return the generator model that can be used to generate fake images."""
def Generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128 * 128 * 3, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.Reshape((128, 128, 3)))
    # downsampling
    model.add(tf.keras.layers.Conv2D(128, 4, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(256, 4, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(512, 4, strides=1, padding='same', kernel_initializer='he_normal',
                                              use_bias=False))
    model.add(tf.keras.layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))

    model.add(tf.keras.layers.LeakyReLU())
    # upsampling
    model.add(tf.keras.layers.Conv2DTranspose(512, 4, strides=1, padding='same', kernel_initializer='he_normal',
                                              use_bias=False))
    model.add(tf.keras.layers.Conv2DTranspose(512, 4, strides=2, padding='same', kernel_initializer='he_normal',
                                              use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(256, 4, strides=1, padding='same', kernel_initializer='he_normal',
                                              use_bias=False))
    model.add(tf.keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same', kernel_initializer='he_normal',
                                              use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer='he_normal',
                                              use_bias=False))
    model.add(tf.keras.layers.Conv2DTranspose(128, 4, strides=1, padding='same', kernel_initializer='he_normal',
                                              use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2DTranspose(3, 4, strides=1, padding='same', activation='tanh'))

    return model

#assigning the funtion to the generator value and plotting the summary of it
generator = Generator()
generator.summary()

"""This function defines and creates a discriminator model for a GAN (Generative Adversarial Network).
The model is a convolutional neural network with several layers that processes images and outputs a probability indicating whether the input image is real or fake.
The model has several convolutional layers with batch normalization and Leaky ReLU activation, followed by a flattening layer and a dense layer with a sigmoid activation function.
The model takes an image of size (SIZE, SIZE, 3) as input and outputs a probability in the range [0, 1]."""
def Discriminator():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input((SIZE, SIZE, 3)))
    model.add(tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(512, 4, strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

#assigning the funtion to the generator value and plotting the summary of it
discriminator = Discriminator()
discriminator.summary()

#giving the noise a random noise and generating an image of the current noise and plotting it
noise = np.random.normal(-1, 1, (1, 100))
img = generator(noise)
plt.imshow(img[0, :, :, 0])
plt.show()

# Compile before plotting
generator.compile(optimizer='adam', loss='binary_crossentropy')
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

#defining the optimizer
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=.0001,
    clipvalue=1.0,
    decay=1e-8
)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#the function is Calculating generator loss using cross-entropy loss function
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

#the function is Calculating discriminator loss using cross-entropy loss function
def discriminator_loss(fake_output, real_output):
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    return fake_loss + real_loss

"""This function represents one training step in a GAN model.
It calculates the generator loss and discriminator loss using the generated images and real images,
and updates the generator and discriminator weights using these losses. It also returns the losses as a dictionary."""
def train_steps(images):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise)
        fake_output = discriminator(generated_images)
        real_output = discriminator(images)

        gen_loss = generator_loss(fake_output)
        dis_loss = discriminator_loss(fake_output, real_output)

    gradient_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_of_discriminator = disc_tape.gradient(dis_loss, discriminator.trainable_variables)

    optimizer.apply_gradients(zip(gradient_of_generator, generator.trainable_variables))
    optimizer.apply_gradients(zip(gradient_of_discriminator, discriminator.trainable_variables))

    loss = {'gen loss': gen_loss,
            'disc loss': dis_loss}
    return loss, gen_loss, dis_loss

# the function is used to plot generated images using the generator model at a specified epoch
def plot_generated_images(square=5, epochs=0):
    plt.figure(figsize=(10, 10))
    for i in range(square * square):
        if epochs != 0:
            if (i == square // 2):
                plt.title("Generated Image at Epoch:{}\n".format(epochs), fontsize=32, color='black')
        plt.subplot(square, square, i + 1)
        noise = np.random.normal(0, 1, (1, latent_dim))
        img = generator(noise)
        plt.imshow(np.clip((img[0, ...] + 1) / 2, 0, 1))

        plt.xticks([])
        plt.yticks([])
        plt.grid()
        plt.show()

# keep track of loss and generated, "fake" samples
losses = []

"""The function "train" is used to train a GAN model with a given dataset and number of epochs.
It prints the loss values and generates images after certain intervals of epochs.
It also saves the generator and discriminator models at certain intervals."""
def train(epochs, dataset):
    serial = 0
    for epoch in range(epochs):
        if serial % 7 == 0:
            print("the generator generated the next photo after ", serial, "epoches")
            plot_generated_images(2)
            plt.show()
        if serial % 10 == 0:
            generator_path = "drive/MyDrive/newgenerator"
            generator_path += str(serial)
            generator_path += '.h5'
            discriminator_path = "drive/MyDrive/newdiscriminator"
            discriminator_path += str(serial)
            discriminator_path += '.h5'
            generator.save(generator_path)
            discriminator.save(discriminator_path)
        start = time.time()
        print("\nEpoch : {}".format(epoch + 1))
        for images in dataset:
            loss, gen_loss, dis_loss = train_steps(images)
        print(" Time:{}".format(np.round(time.time() - start), 2))
        print("Generator Loss: {} Discriminator Loss: {}".format(loss['gen loss'], loss['disc loss']))
        serial = serial + 1
        # append discriminator loss and generator loss
        losses.append((dis_loss, gen_loss))


train(100, dataset)

# showing the scale of losses of generator and discriminator through epochs
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
plt.show()

# saving the trained models in to my google drive
generator.save('drive/MyDrive/newgeneratorAll.h5')
discriminator.save('drive/MyDrive/newdiscriminatorAll.h5')
