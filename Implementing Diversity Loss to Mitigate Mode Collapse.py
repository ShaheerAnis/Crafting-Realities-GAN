# Define diversity loss function
def diversity_loss(y_true, y_pred):
    # Calculate the L2 norm (Euclidean distance) between generated samples
    l2_norm = tf.norm(tf.math.subtract(y_pred[:, 0], y_pred[:, 1]), ord='euclidean')
    # Scale the L2 norm to avoid dominance over other losses
    scaled_l2_norm = l2_norm * 0.1
    return scaled_l2_norm

# Add diversity loss to the GAN model
gan.add_loss(diversity_loss)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss='binary_crossentropy')
