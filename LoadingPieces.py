import os
from pprint import pprint
from NoteStateBatches import *

class LoadingPieces:    
    def load_pieces(dir_path):
        # create the map to store piece names as keys 
        # and note state batches as values
        piece_nsb_map = {}
        
        for midi_file in os.listdir(dir_path):
            if (midi_file[-4:] != ".mid" and midi_file[-4:] != ".MID"):
                continue
            curr = midi_file[:-4]

            note_state_matrix = NoteStateBatches.midi_piece_to_nsb(os.path.join(dir_path, midi_file))   
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
            pprint(piece_nsb_map[curr])
        return piece_nsb_map