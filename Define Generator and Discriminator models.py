import tensorflow as tf
from tensorflow.keras import layers

# Generator Model
def build_generator(latent_dim, output_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(output_shape, activation='tanh'))
    return model

# Discriminator Model
def build_discriminator(input_shape):
    model = tf.keras.Sequential()
    model.add(layers.Dense(512, input_shape=input_shape, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define latent dimension and image shape
latent_dim = 100
img_shape = (28, 28, 1)  # Example shape for grayscale images

# Build and summarize the models
generator = build_generator(latent_dim, img_shape[0] * img_shape[1] * img_shape[2])
generator.summary()

discriminator = build_discriminator(img_shape[0] * img_shape[1] * img_shape[2])
discriminator.summary()
