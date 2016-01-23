"""Config file for Audio Trainer v1.0"""
import pyaudio

# Index of audio device
DEVICE_INDEX = 2

# TODO
CHUNK = 512

# Format of record
FORMAT = pyaudio.paInt16

# Channel to record
CHANNELS = 1

# Filename of wave outputfile
WAVE_OUTPUT_FILENAME = "output.wav"

# Threshold for silence
THRESHOLD = 2200

# How many silent frames should be captured after sound
FRAMES_AFTER_SOUND = 10

# Tolerance for two frames to be the same
TOLERANCE = 120

# Tolerance multiplier for listening
TOLERANCE_MULTIPLIER = 1

# how many frames without a match are allowed before we assume we are not
# in a model
FRAME_COUNT = 10

# counter for waveoutput in multiple files. first file will get this number
WAVENUMBER = 1

# number of iterations of randomized training of a model. If set to 0 it
# will train the model in the order of the files
TRAINING_ITERATIONS = 10

# folder in which the records are stored
RECORDS_DIR = "records"

# folder in which the models are stored
MODELS_DIR = "models"

# name of the tmp folder
TMP_DIR = "tmp"

# standard minimum number of frames per model. Should be greater 2 or the qualitycheck will fail TODO fix that
MIN_FRAMES = 3
