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

# how many frames without a match are allowed before we assume we are not
# in a model
FRAME_COUNT = 5

# folder in which the records are stored
RECORDS_DIR = "records"

# folder in which the models are stored
MODELS_DIR = "models"

# name of the tmp folder
TMP_DIR = "tmp"

# Set to True to train a model with only FREQUENCY_BANDS_TO_COMPARE frequency Bands
FREQUENCY_BAND_TRAINING = False

# If FREQUENCY_BAND_TRAINING is True only the specified number of frequency bands are dropped
FREQUENCY_BANDS_TO_DROP = 16

# The number of frames that should be skipped after a recognition
SKIP_AFTER_RECOGNITION = 10

# The number of features in a frame; should be a power of two
FEATURES_PER_FRAME = 64

# If True then Cepstrum is used as Features, if False then Spectrum is used
# If False than LIFTERING has no effect
# Remember to recalculate all the models if you change this since a Spectral model 
# can't be recognized if recording extraxt Cepstral Features
CEPSTRUM = False

# Liftering for Cepstrum
# has no effect if CEPSTRUM == False
LIFTERING = False

# How many Features should be dropped by Liftering. Usually 15-20
LIFTERING_NUMBER = 15