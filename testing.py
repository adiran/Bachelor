import wave
import numpy as np
import functions as f
from scipy.fftpack import fft, ifft
import config as conf
import pyaudio
import time
import struct
import copy
import scipy.io.wavfile as scwave
import trainRecorded as train

for i in range(1, 26):
    wf = wave.open("records/microwave/" + str(i) + ".wav")
    number = train.preprocess(wf)
    print("\t\tstart new minimalizeAndCalcTolerance")
    beginning = time.time()
    f.minimalizeAndCalcTolerance(number, 5, 1)
    print("\t\tfinished after " + str(time.time() - beginning))
    print("\t\tstart old minimalizeAndCalcTolerance")
    beginning = time.time()
    f.minimalizeAndCalcTolerance1(number, 5, 1)
    print("\t\tfinished after " + str(time.time() - beginning))

for i in range(1, 26):
    wf = wave.open("records/whistleOneTone/" + str(i) + ".wav")
    number = train.preprocess(wf)
    print("\t\tstart new minimalizeAndCalcTolerance")
    beginning = time.time()
    f.minimalizeAndCalcTolerance(number, 5, 1)
    print("\t\tfinished after " + str(time.time() - beginning))
    print("\t\tstart old minimalizeAndCalcTolerance")
    beginning = time.time()
    f.minimalizeAndCalcTolerance1(number, 5, 1)
    print("\t\tfinished after " + str(time.time() - beginning))

for i in range(1, 26):
    wf = wave.open("records/closeDoor/" + str(i) + ".wav")
    number = train.preprocess(wf)
    print("\t\tstart new minimalizeAndCalcTolerance")
    beginning = time.time()
    f.minimalizeAndCalcTolerance(number, 5, 1)
    print("\t\tfinished after " + str(time.time() - beginning))
    print("\t\tstart old minimalizeAndCalcTolerance")
    beginning = time.time()
    f.minimalizeAndCalcTolerance1(number, 5, 1)
    print("\t\tfinished after " + str(time.time() - beginning))