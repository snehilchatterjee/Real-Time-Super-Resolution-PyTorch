import tensorflow as tf

# Define the variable
rrdb_inputs_scales = tf.Variable(
    tf.constant(value=0.25, dtype=tf.float32, shape=[1, 1, 1, 3]),
    name='rrdb_inputs_scales',
    trainable=True
)

print("\nValues before update:")
print(rrdb_inputs_scales.numpy())


# Simulate some loss function
# For demonstration purposes, let's assume a simple loss function
loss = tf.reduce_mean(tf.square(rrdb_inputs_scales - 0.5))  # Loss with respect to some target value of 0.5

# Create an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Inside a training loop or gradient tape context
with tf.GradientTape() as tape:
    # Compute the loss
    loss = 4 * tf.reduce_mean(tf.square(rrdb_inputs_scales - 0.5))
    
    # Compute gradients
    gradients = tape.gradient(loss, rrdb_inputs_scales)
    
# Apply gradients to update the variable
optimizer.apply_gradients(zip([gradients], [rrdb_inputs_scales]))

# Display values after update
print("\nValues after update:")
print(rrdb_inputs_scales.numpy())
