import tensorflow as tf
from tensorflow.python.ops import math_ops

def input_kernel(input_data, midi_low, midi_high, time_init):
    # Capture the input data dimensions
    batch_size = tf.compat.v1.shape(input_data)[0]
    num_notes = input_data.get_shape()[1]
    num_timesteps = tf.compat.v1.shape(input_data)[2]

    # MIDI Note Number (function of the note index)
    # Squeeze - removes dimensions of size 1 from tensor
    midi_indices = tf.squeeze(tf.range(start=midi_low, limit=midi_high + 1, delta=1))

    # Not sure what this does really
    x_midi = tf.ones((batch_size, num_timesteps, 1, num_notes)) * tf.cast(midi_indices, dtype=tf.float32)

    # Transpose the x_midi into dimensions 0 3 1 2
    x_midi = tf.transpose(x_midi, perm=[0, 3, 1, 2])

    # Part Pitch Class
    # Squeeze - removes all dimensions of size 3 from x_midi % 12
    midi_pitchclass = tf.squeeze(x_midi % 12, axis=3)
    x_pitch_class = tf.one_hot(tf.cast(midi_pitchclass, dtype=tf.uint8), depth=12)

    # Part of the Previous Vicinity
    input_flatten = tf.transpose(input_data, perm=[0, 2, 1, 3])
    input_flatten = tf.reshape(input_flatten, [batch_size * num_timesteps, num_notes, 2])
    input_flatten_p = tf.slice(input_flatten, [0, 0, 0], size=[-1, -1, 1])
    input_flatten_a = tf.slice(input_flatten, [0, 0, 1], size=[-1, -1, 1])

    # Reverse Identity Kernel
    filt_vicinity = tf.expand_dims(tf.eye(25), axis=1)

    # 1D convolutional filter for each play and articulate arrays
    vicinity_p = tf.nn.conv1d(input_flatten_p, filt_vicinity, stride=1, padding='SAME')
    vicinity_a = tf.nn.conv1d(input_flatten_a, filt_vicinity, stride=1, padding='SAME')
    
    # Concatenate Back Together and Restack such that play-articulate numbers alternate
    vicinity = tf.stack([vicinity_p, vicinity_a], axis=3)
    vicinity = tf.unstack(vicinity, axis=2)
    vicinity = tf.concat(vicinity, axis=2)

    # Reshape by major dimensions, THEN swap axes
    x_vicinity = tf.reshape(vicinity, shape=[batch_size, num_timesteps, num_notes, 50])
    x_vicinity = tf.transpose(x_vicinity, perm=[0, 2, 1, 3])

    # Part Previous Context
    input_flatten_p_bool = tf.minimum(input_flatten_p, 1)
    # 1 if note is played and 0 if not

    # Kernel
    filt_context = tf.expand_dims(tf.tile(tf.eye(12), multiples=[(num_notes // 12) * 2, 1]), axis=1)

    context = tf.nn.conv1d(input_flatten_p_bool, filt_context, stride=1, padding="SAME")
    x_context = tf.reshape(context, shape=[batch_size, num_timesteps, num_notes, 12])
    x_context = tf.transpose(x_context, perm=[0,2,1,3])

    # beat (function of time axis index plus time_init value)
    time_indices = tf.range(time_init, num_timesteps + time_init)
    x_time = tf.reshape(tf.tile(time_indices, multiples=[batch_size * num_notes]), shape=[batch_size, num_notes, num_timesteps, 1])
    x_beat = tf.cast(tf.concat([x_time % 2, x_time // 2 % 2, x_time // 4 % 2, x_time // 8 % 2], axis=-1), dtype=tf.float32)

    x_zero = tf.zeros([batch_size, num_notes, num_timesteps, 1])

    # Final Vector
    note_state_expand = tf.concat([x_midi, x_pitch_class, x_vicinity, x_context, x_beat, x_zero], axis=-1)
    return note_state_expand

def lstm_timewise_training_layer(input_data, state_init, output_keep_prob=1.0):

    # LSTM time-wise
    # This section is the "Model LSTM-TimeAxis" block and will run a number of LSTM cells
    # over the time axis. Every note and sample in batch will be run in parallel with the
    # same LSTM weights.

    # batch_size and num_timesteps are variable length
    batch_size = tf.shape(input_data)[0]
    num_notes = input_data.get_shape()[1].value
    num_timesteps = tf.shape(input_data)[2]
    input_size = input_data.get_shape()[3].value
    
    num_layers = len(state_init)

    # Flatten input
    input_flatten = tf.reshape(input_data, shape=[batch_size * num_notes, num_timesteps, input_size])

    # Generate cell list of length specified by initial state
    cell_list = []
    num_states = []
    for h in range(num_layers):
        num_states.append(state_init[h][0].get_shape()[1].value)
        lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_states[h], forget_bias=1.0, state_is_tuple=True, activation=math_ops.tanh, reuse=None)
        lstm_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=output_keep_prob)
        cell_list.append(lstm_cell)
    
    # Instantiate multi-layer Time-Wise Cell
    multi_lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cell_list, state_is_tuple=True)

    # Run through LSTM time steps and generate time-wise sequence of outputs
    output_flat, state_out = tf.compat.v1.nn.dynamic_rnn(cell=multi_lstm_cell, inputs=input_flatten, initial_state=state_init, dtype=tf.float32)

    output = tf.reshape(output_flat, shape=[batch_size, num_notes, num_timesteps, num_states[-1]])
    return output, state_out