import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Visualize the model architecture
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
