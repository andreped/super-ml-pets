"""
Implementations of RL model architectures
"""

import tensorflow as tf
from tensorflow import keras

def agent(state_shape, action_shape):
	""" The agent maps X-states to Y-actions
	e.g. The neural network output is [.1, .7, .1, .3]
	The highest value 0.7 is the Q-Value.
	The index of the highest action (0.7) is action #1.
	"""
	learning_rate = 0.001
	init = tf.keras.initializers.HeUniform()
	model = keras.Sequential()
	model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
	model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
	model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
	model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate))  # , metrics=['accuracy'])
	return model
