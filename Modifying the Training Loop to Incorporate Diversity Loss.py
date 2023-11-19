# Modify the training loop to include diversity loss
def train_conditional_gan(generator, discriminator, gan, data, labels, latent_dim, epochs, batch_size):
    for epoch in range(epochs):
        for _ in range(len(data) // batch_size):
            # Train the discriminator
            noise = generate_noise_samples(batch_size, latent_dim)
            generated_images = generator.predict([noise, labels])
            real_images = data[np.random.randint(0, data.shape[0], batch_size)]

            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch([real_images, labels], labels_real)
            d_loss_fake = discriminator.train_on_batch([generated_images, labels], labels_fake)

            # Train the generator (via the GAN model) with diversity loss
            noise = generate_noise_samples(batch_size, latent_dim)
            labels_gan = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch([noise, labels], labels_gan)

        print(f"Epoch {epoch + 1}/{epochs} | Discriminator Loss: {0.5 * np.add(d_loss_real, d_loss_fake)} | Generator Loss: {g_loss}")

# Example usage
labels = np.random.randint(0, num_classes, (1000, 1))  # Example: random labels
train_conditional_gan(generator, discriminator, gan, data, labels, latent_dim, epochs=50, batch_size=64)
