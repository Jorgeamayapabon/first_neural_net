import tensorflow as tf
import numpy as np

celsiud = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenhiet = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

hide1 = tf.keras.layers.Dense(units=3, input_shape=[1])
hide2 = tf.keras.layers.Dense(units=3)
output = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([hide1, hide2, output])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsiud, fahrenhiet, epochs=500, verbose=False)
