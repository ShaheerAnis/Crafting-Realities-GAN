# Additional imports
from tensorflow.keras.layers import Input, Concatenate

# New Generator for Conditional GAN
def build_conditional_generator(latent_dim, output_shape, num_classes):
    # Latent input
    noise = Input(shape=(latent_dim,))
    # Conditional input
    label = Input(shape=(1,), dtype='int32')

    # Embedding layer for labels
    label_embedding = layers.Embedding(num_classes, latent_dim)(label)
    label_embedding = layers.Flatten()(label_embedding)

    # Concatenate noise and label embeddings
    combined_input = layers.Concatenate()([noise, label_embedding])

    # Generator layers
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim + num_classes, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(output_shape, activation='tanh'))

    # Model with combined input
    generated_image = model(combined_input)
    return tf.keras.Model([noise, label], generated_image)

# New Discriminator for Conditional GAN
def build_conditional_discriminator(input_shape, num_classes):
    # Image input
    image = Input(shape=input_shape)
    # Conditional input
    label = Input(shape=(1,), dtype='int32')

    # Embedding layer for labels
    label_embedding = layers.Embedding(num_classes, np.prod(input_shape))(label)
    label_embedding = layers.Flatten()(label_embedding)

    # Flatten the image
    flat_image = layers.Flatten()(image)

    # Concatenate flattened image and label embeddings
    combined_input = layers.Concatenate()([flat_image, label_embedding])

    # Discriminator layers
    model = tf.keras.Sequential()
    model.add(layers.Dense(512, input_dim=np.prod(input_shape) + num_classes, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Model with combined input
    validity = model(combined_input)
    return tf.keras.Model([image, label], validity)
