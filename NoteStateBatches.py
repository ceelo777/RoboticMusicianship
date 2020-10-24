import midi
import numpy as np

# Lower bound for notes possible
lower_bound = 24

# Upper bound for notes possible
upper_bound = 102

# Define the range of bounds
bound_span = upper_bound - lower_bound 

def midi_piece_to_nsb(midi_file):
    # Read in the pattern for the inputted MIDI file
    pattern = midi.read_midifile(midi_file)

    # Set the time left for each track's current event to the first event's tick
    time_left = []
    for track in pattern:
        time_left.append(track[0].tick)

    # Set the positions for each track to 0
    track_positions = []
    for track in pattern:
        track_positions.append(0)
    
    # Initialize the note state matrix
    note_state_matrix = []

    # Current time for the whole pattern        
    curr_time = 0

    # Create a current state array
    curr_state = []

    # Add [0, 0] to indicate pitch and articulation
    # for every note within bounds
    for note in range(bound_span):
        curr_state.append([0, 0])
    
    # Add this initial current state to our note state matrix
    note_state_matrix.append(curr_state)

    # Set condition to true
    flag = True

    # Begin iterating through the tracks
    while (flag):

        # If a note boundary is crossed for the note state matrix
        if (curr_time % (pattern.resolution / 4) == pattern.resolution / 8):                

            # Set the previous state to the current state
            prev_state = curr_state

            # Create new state with holding note pitches being default
            curr_state = []
            for note in range(bound_span):
                curr_state.append([prev_state[note][0], 0])
            
            # Add to the note state matrix
            note_state_matrix.append(curr_state)

        # Iterate through all the tracks by index
        for i in range(len(pattern)):

            # If flag is set to false, break
            if (flag == False):
                break

            # Advance through all the notes in the track
            # where the time held is just 0
            while (time_left[i] == 0):
                track = pattern[i]
                position = track_positions[i]
                event = track[position]

                # If a note event is reached
                if (isinstance(event, midi.NoteEvent)):

                    # If the pitch is out of the bounds, skip it
                    if (event.pitch < lower_bound or event.pitch >= upper_bound):
                        pass                    
                    # Otherwise
                    else:
                        
                        # If a note off event is reached or velocity is 0 set to 0
                        if (isinstance(event, midi.NoteOffEvent) or event.velocity == 0):
                            curr_state[event.pitch - lower_bound] = [0, 0]
                        # Otherwise set to 1
                        else:
                            curr_state[event.pitch - lower_bound] = [1, 1]
                
                # If a time signature event is reached
                elif (isinstance(event, midi.TimeSignatureEvent)):
                    # break out of non-4 time signatures
                    if (event.numerator not in (2, 4)):
                        output = note_state_matrix
                        flag = False
                        break

                # Try setting the time left for the next track
                # to the next tick and increasing position by one
                try:
                    time_left[i] = track[position + 1].tick
                    track_positions[i] += 1
                
                # If index error reached, error out with index
                # track is finished
                except IndexError:
                    time_left[i] = None

            # If the track is not done, decrease the time for
            # the current event by 1 tick
            if time_left[i] != None:     
                time_left[i] -= 1

            # Check if all tracks are finished
            done = True
            for time in time_left:
                if (time != None):
                    done = False                
            if (done):
                flag = False
                break                                    

            # increase the current time for the pattern by 1
            curr_time += 1
            # print(curr_time)
            # print(time_left)                
    # print(curr_time)
    return note_state_matrix        