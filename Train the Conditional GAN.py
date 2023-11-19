# Modify the Discriminator for conditional input
discriminator = build_conditional_discriminator(img_shape[0] * img_shape[1] * img_shape[2], num_classes)
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# GAN Model for Conditional GAN
gan_input = tf.keras.Input(shape=(latent_dim,))
gan_label = tf.keras.Input(shape=(1,), dtype='int32')
generated_image = generator([gan_input, gan_label])
gan_output = discriminator([generated_image, gan_label])

# Compile the GAN for Conditional GAN
gan = tf.keras.Model([gan_input, gan_label], gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss='binary_crossentropy')
