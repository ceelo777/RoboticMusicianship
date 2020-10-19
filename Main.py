# Import our function
from LoadingPieces import *

# Import required packages
import os

# Set the current directory
directory = os.getcwd()

# Set the directory to all the music files
music_directory = directory + "/MIDI_Files"

# Load all the files from the music directory
training_pieces = LoadingPieces.load_pieces(music_directory)

# print(training_pieces)