import numpy as np

# Function to generate random noise for the generator
def generate_noise_samples(n_samples, latent_dim):
    return np.random.normal(0, 1, size=(n_samples, latent_dim))

# Function to train the GAN
def train_gan(generator, discriminator, gan, data, latent_dim, epochs, batch_size):
    for epoch in range(epochs):
        for _ in range(len(data) // batch_size):
            # Train the discriminator
            noise = generate_noise_samples(batch_size, latent_dim)
            generated_images = generator.predict(noise)
            real_images = data[np.random.randint(0, data.shape[0], batch_size)]

            labels_real = np.ones((batch_size, 1))
            labels_fake = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(real_images, labels_real)
            d_loss_fake = discriminator.train_on_batch(generated_images, labels_fake)

            # Train the generator (via the GAN model)
            noise = generate_noise_samples(batch_size, latent_dim)
            labels_gan = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise, labels_gan)

        print(f"Epoch {epoch + 1}/{epochs} | Discriminator Loss: {0.5 * np.add(d_loss_real, d_loss_fake)} | Generator Loss: {g_loss}")

# Example usage
data = np.random.random((1000, img_shape[0] * img_shape[1] * img_shape[2]))  # Example: random data
train_gan(generator, discriminator, gan, data, latent_dim, epochs=50, batch_size=64)
