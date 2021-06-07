# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:42:22 2021

@author: Fady
Credit is also given to J. Brownlee for his article "How to Develop an Auxiliary Classifier GAN (AC-GAN) From Scratch with Keras"
"""
# Importing necessary python libraries
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot
import tensorflow as tf

# Setting proper number of GPUs for model development
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Creating the discriminator component 
def define_discriminator(in_shape=(28,28,1), n_classes=10):
	# Creating weights
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=in_shape)
	# Downsampling image to 14 x 14
	fe = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	
	fe = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# Downsampling image to 7 x 7
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	
	fe = Conv2D(256, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# Reshaping Feature Maps
	fe = Flatten()(fe)
	# Creating Output for Real/Fake Classification
	out1 = Dense(1, activation='sigmoid')(fe)
	# Creating Output for 10-Class Label Classification
	out2 = Dense(n_classes, activation='softmax')(fe)
	# Building Model default from Keras
	model = Model(in_image, [out1, out2])
	# Compile Model with Adam Optimizer and Loss Functions
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
	return model

# Create Generator Component
def define_generator(latent_dim, n_classes=10):
	# Define intial weights
	init = RandomNormal(stddev=0.02)

	in_label = Input(shape=(1,))
	# Create Embedding Layer for input of different categories - convert class label to feature map
	li = Embedding(n_classes, 50)(in_label)
	
	n_nodes = 7 * 7
	li = Dense(n_nodes, kernel_initializer=init)(li)
	# Reshape class label into additional feature map
	li = Reshape((7, 7, 1))(li)

	in_lat = Input(shape=(latent_dim,))
	# Develop an initial 7x7 image based on random noise vector input
	n_nodes = 384 * 7 * 7
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
	gen = Activation('relu')(gen)
	gen = Reshape((7, 7, 384))(gen)
	# Merge class label feature map and 7x7 feature map
	merge = Concatenate()([gen, li])
	# Upsampling merged image to 14x14
	gen = Conv2DTranspose(192, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(merge)
	gen = BatchNormalization()(gen)
	gen = Activation('relu')(gen)
	# Upsampling to 28x28 output image
	gen = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	out_layer = Activation('tanh')(gen)
	# Define the final generator model
	model = Model([in_lat, in_label], out_layer)
	return model

# Create full AC-GAN model
def define_gan(g_model, d_model):
	# Prevent weights in discriminator from being trained
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# Feed generated images into discriminator
	gan_output = d_model(g_model.output)
	# Define discriminator to take noise + generated image and output 2 classifications
	model = Model(g_model.input, gan_output)
	# Build model with Adam optimizer
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
	return model


def load_real_samples():
	# Load Keras Fashion-MNIST dataset
	(trainX, trainy), (_, _) = load_data()
	# Add additional chanels upto RGB
	X = expand_dims(trainX, axis=-1)
	
	X = X.astype('float32')
	# Scaling to [-1, 1] for use with tanh function
	X = (X - 127.5) / 127.5
	print(X.shape, trainy.shape)
	return [X, trainy]


def generate_real_samples(dataset, n_samples):
	# Divide dataset into images and labels
	images, labels = dataset
	# Create random noise vector
	ix = randint(0, images.shape[0], n_samples)
	# Select images and labels to scramble
	X, labels = images[ix], labels[ix]
	# Build class labels
	y = ones((n_samples, 1))
	return [X, labels], y

# Create additional input for generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	# Make points in latent space
	x_input = randn(latent_dim * n_samples)
	# Reshape input 
	z_input = x_input.reshape(n_samples, latent_dim)
	# Create new labels for images
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
	# Generate starting points in latent space with generated image
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# Create output image
	images = generator.predict([z_input, labels_input])
	# Build Class Labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

def summarize_performance(step, g_model, latent_dim, n_samples=100):
	# Generate the fake images
	[X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# Scale properly for pyplot
	X = (X + 1) / 2.0
	# Show images
	for i in range(100):
		# Create figure of subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# Plot the pixel image
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot
	filename1 = 'generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save model
	filename2 = 'model_%04d.h5' % (step+1)
	g_model.save(filename2)
	print('Saved: %s and %s' % (filename1, filename2))

# Train entire AC-GAN
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=64):
	# Calculate batches per epoch
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# Calculate total training iterations
	n_steps = bat_per_epo * n_epochs
	
	half_batch = int(n_batch / 2)
	# Number epochs
	for i in range(n_steps):
		# Grab real images
		[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
		# Train discriminator on real images
		_,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
		# Create fake images
		[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# Train discriminator on fake images
		_,d_f,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
		# Prepate input for generator
		[z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
		# Build labels for fake images
		y_gan = ones((n_batch, 1))
		# Update generator error dependent on discriminator error
		_,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
		# Show loss
		print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f,d_f2, g_1,g_2))
		# Evaluate performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, latent_dim)

# Set size of latent dimension
latent_dim = 100
# Build discriminator
discriminator = define_discriminator()
# Build generator
generator = define_generator(latent_dim)
# Build the AC-GAN
gan_model = define_gan(generator, discriminator)
# Load real image data
dataset = load_real_samples()
# Train AC-GAN
train(generator, discriminator, gan_model, dataset, latent_dim)
    
