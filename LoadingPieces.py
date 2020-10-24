import os
import numpy as np
import random

from pprint import pprint
from NoteStateBatches import *
from Converting import *

div_len = 16

def load_pieces(dir_path):
    # create the map to store piece names as keys 
    # and note state batches as values
    piece_nsb_map = {}
    
    for midi_file in os.listdir(dir_path):
        if (midi_file[-4:] != ".mid" and midi_file[-4:] != ".MID"):
            continue
        curr = midi_file[:-4]

        print(midi_file)     
        try:        
            note_state_matrix = midi_piece_to_nsb(os.path.join(dir_path, midi_file))  
        except: 
            print("Skip bad file: ", curr)
            note_state_matrix = [] 
        print("Finished processing a MIDI File.")

        if (len(note_state_matrix) < 256):
            continue
        # note_state_matrix = NoteStateBatches.midiToNoteStateMatrix(os.path.join(dir_path, midi_file))        
        """                                   
        try: 
            # Convert this to Note State Batch (call right here when written)
            note_state_matrix = NoteStateBatches.midi_piece_to_nsb(os.path.join(dir_path, midi_file))                
        except: 
            # Set value to empty if a bad file is reached
            print("Bad File: ", midi_file)
            note_state_matrix = []
        """

        # Add to the pieces map
        piece_nsb_map[curr] = note_state_matrix
        # print("Loaded ", curr)
        # pprint(piece_nsb_map[curr])
    return piece_nsb_map    

def get_piece_segment(pieces, time_steps):
    # Pick a random piece in the pieces dictionary
    piece_nsb = random.choice(list(pieces.values()))

    # Collect a segment in the form of a note state matrix    
    start = random.randrange(0, len(piece_nsb) - time_steps, div_len)
    nsb_segment = piece_nsb[start : start + time_steps]

    # Get the music form of the note state matrix
    # pprint(nsb_segment)
    piece_segment = nsb_to_input(nsb_segment)

    # Return both the piece segment and its corresponding nsb
    return piece_segment, nsb_segment


def get_piece_batch(pieces, batch_size, time_steps):
    piece_segments, nsb_segments = zip(*[get_piece_segment(pieces, time_steps) for _ in range(batch_size)])
    return np.array(piece_segments), np.array(nsb_segments)