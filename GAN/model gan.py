# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:06:26 2021

@author: Fady
Credit is also given to J. Brownlee for his article "How to Develop an Auxiliary Classifier GAN (AC-GAN) From Scratch with Keras"
"""

# Import necessary Python libraries
from math import sqrt
from numpy import asarray
from numpy.random import randn
from keras.models import load_model
from matplotlib import pyplot

# Generate input for generator 
def generate_latent_points(latent_dim, n_samples, n_class):
    # Get random noise vector
    x_input = randn(latent_dim * n_samples)
    # Reshape input into numerical input for generator
    z_input = x_input.reshape(n_samples, latent_dim)
    # Create labels
    labels = asarray([n_class for _ in range(n_samples)])
    return [z_input, labels]

# Save AC-GAN generated images
def save_plot(examples, n_examples,n_class):
    # Turn off plotting
    pyplot.ioff()
    for i in range(n_examples):
        # Save generated images
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
        pyplot.savefig(str(n_class)+"_"+str(i))



model = load_model('model_93700.h5')
latent_dim = 100
n_examples = 100 
n_class = 0 
# Generate latent points (input) for generator
latent_points, labels = generate_latent_points(latent_dim, n_examples, n_class)
# Generate fake images
X  = model.predict([latent_points, labels])
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# Save images
save_plot(X, n_examples,n_class)