# Import our function
from LoadingPieces import *
from Layers import input_kernel, lstm_timewise_training_layer

# Import required packages
import os
import random
import numpy as np
import tensorflow as tf

# Set the current directory
directory = os.getcwd()

# Set the directory to all the music files
music_directory = directory + "/MIDI_Files/Piano_Midi/"

# Load all the files from the music directories
# MIDI_Directories = ["albeniz", "beeth", "borodin", "brahms", "burgm", "Chopin", "debussy", "granados", "grieg", "haydn", "liszt", "mendelssohn", "mozart", "muss", "schubert", "schumann", "tschai"]
MIDI_Directories = ["beeth"]
training_pieces = {}
for i in range(len(MIDI_Directories)):
    curr = music_directory + MIDI_Directories[i]
    print(curr)
    training_pieces.update(load_pieces(curr))

# Set aside random set of pieces for validation set
validation_count = 0
validation_set = {}
for i in range(validation_count):
    random_key = random.choice(list(training_pieces.keys()))
    validation_set[random_key] = training_pieces.pop(random_key)

print(list(training_pieces.keys()))
print(list(validation_set.keys()))

batch_size = 15
time_steps = 128
# Call Get Piece Batch
_, sample_state = get_piece_batch(training_pieces, batch_size, time_steps)
sample_state = np.array(sample_state)
sample_state = np.swapaxes(sample_state, axis1=1, axis2=2)
print('State Input Batch Shape = ', sample_state.shape)

# Begin building the model graph for training
tf.compat.v1.reset_default_graph()

# Capture the number of notes from the sample
num_notes = sample_state.shape[1]

# Placeholders for the graph input
tf.compat.v1.disable_eager_execution()
note_state_batch = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_notes, None, 2])
initial_time = tf.compat.v1.placeholder(dtype=tf.int32, shape=())

# Generate expanded tensor from the batch of note state matrices
note_state_expand = input_kernel(note_state_batch, midi_low=24, midi_high=101, time_init=initial_time)
print("Note State Expand Shape: ", note_state_expand.get_shape())

# LSTM Time Wise Training Graph
num_t_units = [200, 200]
output_keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, shape=())

# Generate Initial State (at t = 0) placeholder
timewise_state = []
for i in range(len(num_t_units)):
    timewise_c = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_t_units[i]]) # None = batch_size * num_notes
    timewise_h = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_t_units[i]])
    timewise_state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(timewise_h, timewise_c))

timewise_state = tuple(timewise_state)

timewise_out, timewise_state_out = lstm_timewise_training_layer(input_data=note_state_expand, state_init=timewise_state, output_keep_prob=output_keep_prob)
print("Time-wise output shape = ", timewise_out.get_shape())