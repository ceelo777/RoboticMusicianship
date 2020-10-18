import os

class LoadingPieces:
    def __init__(self, time_steps):        
        self.time_steps = time_steps

    def load_pieces(self, dir_path, time_steps):
        pieces = {}
        
        for midi_file in os.listdir(dir_path):
            if (midi_file[-4:] != ".mid" and midi_file[-4:] != ".MID"):
                continue
            curr = midi_file[:-4]
            # Convert this to Note State Batch (call right here when written)
            