import tensorflow as tf
from tensorflow.python.ops import math_ops


def input_kernel(input_data, midi_low, midi_high, time_init):
    # Capture the input data dimensions
    batch_size = tf.compat.v1.shape(input_data)[0]
    num_notes = input_data.get_shape()[1]
    num_timesteps = tf.compat.v1.shape(input_data)[2]

    # MIDI Note Number (function of the note index)
    # Squeeze - removes dimensions of size 1 from tensor
    midi_indices = tf.squeeze(
        tf.range(start=midi_low, limit=midi_high + 1, delta=1))

    # Not sure what this does really
    x_midi = tf.ones((batch_size, num_timesteps, 1, num_notes)
                     ) * tf.cast(midi_indices, dtype=tf.float32)

    # Transpose the x_midi into dimensions 0 3 1 2
    x_midi = tf.transpose(x_midi, perm=[0, 3, 1, 2])

    # Part Pitch Class
    # Squeeze - removes all dimensions of size 3 from x_midi % 12
    midi_pitchclass = tf.squeeze(x_midi % 12, axis=3)
    x_pitch_class = tf.one_hot(
        tf.cast(midi_pitchclass, dtype=tf.uint8), depth=12)

    # Part of the Previous Vicinity
    input_flatten = tf.transpose(input_data, perm=[0, 2, 1, 3])
    input_flatten = tf.reshape(
        input_flatten, [batch_size * num_timesteps, num_notes, 2])
    input_flatten_p = tf.slice(input_flatten, [0, 0, 0], size=[-1, -1, 1])
    input_flatten_a = tf.slice(input_flatten, [0, 0, 1], size=[-1, -1, 1])

    # Reverse Identity Kernel
    filt_vicinity = tf.expand_dims(tf.eye(25), axis=1)

    # 1D convolutional filter for each play and articulate arrays
    vicinity_p = tf.nn.conv1d(
        input_flatten_p, filt_vicinity, stride=1, padding='SAME')
    vicinity_a = tf.nn.conv1d(
        input_flatten_a, filt_vicinity, stride=1, padding='SAME')

    # Concatenate Back Together and Restack such that play-articulate numbers alternate
    vicinity = tf.stack([vicinity_p, vicinity_a], axis=3)
    vicinity = tf.unstack(vicinity, axis=2)
    vicinity = tf.concat(vicinity, axis=2)

    # Reshape by major dimensions, THEN swap axes
    x_vicinity = tf.reshape(
        vicinity, shape=[batch_size, num_timesteps, num_notes, 50])
    x_vicinity = tf.transpose(x_vicinity, perm=[0, 2, 1, 3])

    # Part Previous Context
    input_flatten_p_bool = tf.minimum(input_flatten_p, 1)
    # 1 if note is played and 0 if not

    # Kernel
    filt_context = tf.expand_dims(
        tf.tile(tf.eye(12), multiples=[(num_notes // 12) * 2, 1]), axis=1)

    context = tf.nn.conv1d(input_flatten_p_bool,
                           filt_context, stride=1, padding="SAME")
    x_context = tf.reshape(
        context, shape=[batch_size, num_timesteps, num_notes, 12])
    x_context = tf.transpose(x_context, perm=[0, 2, 1, 3])

    # beat (function of time axis index plus time_init value)
    time_indices = tf.range(time_init, num_timesteps + time_init)
    x_time = tf.reshape(tf.tile(time_indices, multiples=[
                        batch_size * num_notes]), shape=[batch_size, num_notes, num_timesteps, 1])
    x_beat = tf.cast(tf.concat([x_time % 2, x_time // 2 % 2, x_time // 4 %
                                2, x_time // 8 % 2], axis=-1), dtype=tf.float32)

    x_zero = tf.zeros([batch_size, num_notes, num_timesteps, 1])

    # Final Vector
    note_state_expand = tf.concat(
        [x_midi, x_pitch_class, x_vicinity, x_context, x_beat, x_zero], axis=-1)
    return note_state_expand


def lstm_timewise_training_layer(input_data, state_init, output_keep_prob=1.0):

    # LSTM time-wise
    # This section is the "Model LSTM-TimeAxis" block and will run a number of LSTM cells
    # over the time axis. Every note and sample in batch will be run in parallel with the
    # same LSTM weights.

    # batch_size and num_timesteps are variable length
    batch_size = tf.shape(input_data)[0]
    num_notes = input_data.get_shape()[1]
    num_timesteps = tf.shape(input_data)[2]
    input_size = input_data.get_shape()[3]

    num_layers = len(state_init)

    # Flatten input
    input_flatten = tf.reshape(
        input_data, shape=[batch_size * num_notes, num_timesteps, input_size])

    # Generate cell list of length specified by initial state
    cell_list = []
    num_states = []
    for h in range(num_layers):
        num_states.append(state_init[h][0].get_shape()[1])
        lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
            num_units=num_states[h], forget_bias=1.0, state_is_tuple=True, activation=math_ops.tanh, reuse=None)
        lstm_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=output_keep_prob)
        cell_list.append(lstm_cell)

    # Instantiate multi-layer Time-Wise Cell
    multi_lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
        cell_list, state_is_tuple=True)

    # Run through LSTM time steps and generate time-wise sequence of outputs
    output_flat, state_out = tf.compat.v1.nn.dynamic_rnn(
        cell=multi_lstm_cell, inputs=input_flatten, initial_state=state_init, dtype=tf.float32)

    output = tf.reshape(output_flat, shape=[
                        batch_size, num_notes, num_timesteps, num_states[-1]])
    return output, state_out


def lstm_notewise_layer(input_data, state_init, output_keep_prob=1.0):
    batch_size = tf.shape(input_data)[0]
    num_notes = input_data.get_shape()[1]
    num_timesteps = tf.shape(input_data)[2]
    input_size = input_data.get_shape()[3]

    num_layers = len(state_init)

    notewise_in = tf.transpose(input_data, perm=[0, 2, 1, 3])
    notewise_in = tf.reshape(
        notewise_in, shape=[batch_size * num_timesteps, num_notes, input_size])

    # Generate LSTM cell list of length specified by initial state
    # each layer has a dropout mask
    cell_list = []
    num_states = []
    for h in range(num_layers):
        num_states.append(state_init[h][0].get_shape()[1])
        lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(
            num_units=num_states[h], forget_bias=1.0, state_is_tuple=True, activation=math_ops.tanh, reuse=None)
        lstm_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=output_keep_prob)
        cell_list.append(lstm_cell)

    # Instantiate multi-layer Time-Wise Cell
    multi_lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
        cell_list, state_is_tuple=True)

    # For this LSTM cell, we cannot use tf.nn.dynamic_rnn because samples have to be generated and fed back to for subsequent notes
    # need to feed the generated output for note 'n-1' into the generation of note 'n'
    # Will use 'for' loop and call the LSTM cell each for each note step

    # Set values for initial LSTM state and sampled note. Zero notes always played below bottom note.
    h_state = state_init
    p_gen_n = tf.zeros([batch_size*num_timesteps, 1])
    a_gen_n = tf.zeros([batch_size*num_timesteps, 1])

    y_list = []
    note_gen_list = []

    # Run through notes for note-wiseLSTM to obtain P(va(n)) | va(<n))
    for n in range(num_notes):
        # concatenate previously sampled note play-articulate-combo with timewise output
        # feed back both 'play' and 'articulate' components (articulate component is the masked version)
        cell_inputs = tf.concat([notewise_in[:, n, :], tf.cast(
            p_gen_n, tf.float32), tf.cast(a_gen_n, tf.float32)], axis=-1)

        # print("Cell inputs shape = ", cell_inputs.get_shape())"

        # output shape = [batch_size * num_timesteps, Nfinal]
        h_final_out, h_state = multi_lstm_cell(
            inputs=cell_inputs, state=h_state)

        #print('h_final_out shape = ', h_final_out.get_shape())
        #print('h_state len = ', len(h_state))

        # Fully Connected Layer to generate 2 outputs that correspond to [logit(p=1), logit(a=1)]
        # keeping it in logit form for convenience of Tensorflow functions (logit=inverse sigmoid = 1:1 mapping with probability)
        y_n = tf.layers.dense(inputs=h_final_out, units=2, activation=None)
        #print('y_n shape = ', y_n.get_shape())

        # sample note play/articulation using Probability of event = sigmoid(y_n)
        note_gen_n = tf.distributions.Bernoulli(logits=y_n).sample()
        #print('note_gen_n original shape = ', note_gen_n.get_shape())

        """
        Network should never generate an articulation with a 'no play' note.  
        The  Midi-to-Matrix function never generates this condition, so during music generation, feeding this condition into                       the next batch creates inputs that the model has never seen.
        """
        # zero out all articulation where the same note at the same time step of the same batch is NOT played
        p_gen_n = tf.slice(note_gen_n, [0, 0], [-1, 1])
        a_gen_n = tf.slice(note_gen_n, [0, 1], [-1, 1])
        # if a given note is not played (=0), automatically set articulate to zero.
        a_gen_n = tf.multiply(p_gen_n, a_gen_n)
        note_gen_n = tf.concat([p_gen_n, a_gen_n], axis=1)  # concatenate

        #print('note_gen_n final shape = ', note_gen_n.get_shape())

        # Reshape the 1st dimension back into batch and timesteps dimensions
        y_n_unflat = tf.reshape(y_n, shape=[batch_size, num_timesteps, 2])
        note_gen_n_unflat = tf.reshape(
            note_gen_n, shape=[batch_size, num_timesteps, 2])

        #print('note_gen_n shape = ', note_gen_n.get_shape())

        # Append to notewise list
        y_list.append(y_n_unflat)
        note_gen_list.append(note_gen_n_unflat)

    # Convert output list to a Tensor
    y_out = tf.stack(y_list, axis=1)
    note_gen_out = tf.stack(note_gen_list, axis=1)

    return y_out, note_gen_out


def loss_function(Note_State_Batch, y_out):
    """
    Arguments:
        Note State Batch: shape = [batch_size x num_notes x num_timesteps x 2]
        batch of logit(prob=1): shape = [batch_size x num_notes x num_timesteps x 2]

    # This section is the Loss Function Block
    # Note_State_Batch contains the actual binary values played and articulated for each note, at every time step, for every batch
    # Entries in y_out at time step $t$ were generated by entries in Note_State_Batch at time step $t$.  The objective of the model is for 
    entries in y_out at time step $t$ to predict Note_State_Batch at time step $t+1$.  In order to properly align the tensors for the  
    loss function calculation, the last time slice of y_out is removed, and the first time slice of Note_State_Batch is removed.
    """

    # batch_size and num_timesteps are variable length
    batch_size = tf.shape(y_out)[0]
    num_notes = y_out.get_shape()[1].value
    num_timesteps = tf.shape(y_out)[2]

    # Line up y_out  with next-time-step note_state input data
    y_align = tf.slice(y_out, [0, 0, 0, 0], [
                       batch_size, num_notes, num_timesteps-1, 2])
    Note_State_Batch_align = tf.slice(Note_State_Batch, [0, 0, 1, 0], [
                                      batch_size, num_notes, num_timesteps-1, 2])

    #print('Note_State_Batch_align shape = : ', Note_State_Batch_align.get_shape())
    #print('y_align shape = : ', y_align.get_shape())

    # calculate log likelihoods
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y_align, labels=Note_State_Batch_align)

    # if note is not played, mask out loss term for articulation
    cross_entropy_p = cross_entropy[:, :, :, 0]
    cross_entropy_a = cross_entropy[:, :, :,
                                    1] * Note_State_Batch_align[:, :, :, 0]
    cross_entropy = tf.stack([cross_entropy_p, cross_entropy_a], axis=-1)

    # calculate the loss function as defined in the paper
    # negative log-likelihood of batch (factor of 2 for both play and articulate)
    Loss = tf.reduce_mean(cross_entropy) * 2

    # calculate the log-likelihood of notes at a single time step
    Log_likelihood = -Loss*num_notes

    return Loss, Log_likelihood
