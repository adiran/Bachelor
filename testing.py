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

import sys


def printmodel(model):
    print("\nLength: " +str(len(model)) + "\n")
    for i in range(32):
        for j in range(len(model) - 1):
            sys.stdout.write(str(model[j][i]) + "\t")
        print(model[4][i])
    print("\n")

if True:
    for i in range(1, 3):
        wf = wave.open("records/microwave/" + str(i) + ".wav")
        number = train.preprocess((wf, 0))
        print("\t\tstart new minimalizeAndCalcTolerance")
        beginning = time.time()
        model, tolerances = f.minimalizeAndCalcTolerance((number, 5, 1))
        printmodel(model)
        print("Tolerances: " + str(tolerances))
        print("\t\tfinished after " + str(time.time() - beginning))
if True:
    for i in range(1, 3):
        wf = wave.open("records/whistleOneTone/" + str(i) + ".wav")
        number = train.preprocess((wf, 0))
        print("\t\tstart new minimalizeAndCalcTolerance")
        beginning = time.time()
        model, tolerances = f.minimalizeAndCalcTolerance((number, 5, 1))
        printmodel(model)
        print("Tolerances: " + str(tolerances))
        print("\t\tfinished after " + str(time.time() - beginning))
if True:
    for i in range(1, 3):
        wf = wave.open("records/closeDoor/" + str(i) + ".wav")
        number = train.preprocess((wf, 0))
        print("\t\tstart new minimalizeAndCalcTolerance")
        beginning = time.time()
        model, tolerances = f.minimalizeAndCalcTolerance((number, 5, 1))
        printmodel(model)
        print("Tolerances: " + str(tolerances))
        print("\t\tfinished after " + str(time.time() - beginning))

conf.FREQUENCY_BAND_TRAINING = True
print("-------------------------------------------------------------")
if True:
    for i in range(1, 3):
        wf = wave.open("records/microwave/" + str(i) + ".wav")
        number = train.preprocess((wf, 0))
        print("\t\tstart new minimalizeAndCalcTolerance")
        beginning = time.time()
        model, tolerances = f.minimalizeAndCalcTolerance((number, 5, 1))
        #printmodel(model)
        print("Tolerances: " + str(tolerances))
        print("\t\tfinished after " + str(time.time() - beginning))
if True:
    for i in range(1, 3):
        wf = wave.open("records/whistleOneTone/" + str(i) + ".wav")
        number = train.preprocess((wf, 0))
        print("\t\tstart new minimalizeAndCalcTolerance")
        beginning = time.time()
        model, tolerances = f.minimalizeAndCalcTolerance((number, 5, 1))
        #printmodel(model)
        print("Tolerances: " + str(tolerances))
        print("\t\tfinished after " + str(time.time() - beginning))
if True:
    for i in range(1, 3):
        wf = wave.open("records/closeDoor/" + str(i) + ".wav")
        number = train.preprocess((wf, 0))
        print("\t\tstart new minimalizeAndCalcTolerance")
        beginning = time.time()
        model, tolerances = f.minimalizeAndCalcTolerance((number, 5, 1))
        #printmodel(model)
        print("Tolerances: " + str(tolerances))
        print("\t\tfinished after " + str(time.time() - beginning))
