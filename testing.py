import pyaudio
import wave
import time
import sys
import struct
import math
import audioop
import os.path
import random
import config as conf
import numpy as np
import multiprocessing
import copy
import gc
#from memory_profiler import profile
#from guppy import hpy
import pdb

from scipy.fftpack import fft

# import of own scripts
import functions as f
import qualitycheck as q
import interactions



def modelMergeNew(data):
    model, minFrames, maxFrames, name, iteration = data
    f.modelMergeNew(model, minFrames, maxFrames, name, iteration)

#@profile
def preprocess(wf, fileName, wavenumber):
    # check wether the wave file is mono or stereo
    if wf.getnchannels() == 1:
        print("File " + str(fileName) + "/" + str(wavenumber) +
                ".wav found. Processing it now...")
        loops = int(wf.getnframes() / conf.CHUNK)
        framesAfterSound = conf.FRAMES_AFTER_SOUND
        number = []
        switch = True
        for i in range(loops):
            
            framesAsString = wf.readframes(512)
            audioLevel = math.sqrt(
                abs(audioop.avg(framesAsString, 4)))
            #print("AudioLevel of these frames: " + str(audioLevel) + " | THRESHOLD: " + str(conf.THRESHOLD))
            framesSwitch = True
            # if audio level is under conf.THRESHOLD but we had sound in the frames before we capture a few frames more to prevent single missing frames
            # first we check the audio level because framesAfterSound > 0
            # occures way more than aduioLevel <= conf.THRESHOLD
            if audioLevel <= conf.THRESHOLD:
                if framesAfterSound > 0:
                    audioLevel = conf.THRESHOLD + 1
                    framesAfterSound -= 1
                    framesSwitch = False
            # the whole preprocessing
            if audioLevel > conf.THRESHOLD:
                # if framesAfterSound has been decrement but audio level rised over conf.THRESHOLD again we reset framesAfterSound
                # first we check if we decreased framesAfterSound because
                # framesSwitch is True by default
                if framesAfterSound < conf.FRAMES_AFTER_SOUND:
                    if framesSwitch:
                        framesAfterSound = conf.FRAMES_AFTER_SOUND
                if switch:
                    frame = np.fromstring(framesAsString, np.int16)
                    switch = False
                else:
                    frame = np.append(
                        frame, np.fromstring(framesAsString, np.int16))
                    #print("Length frame: " + str(len(frame)))
                    number.append(f.extractFeatures(f.process(frame)))
                    switch = True
            del framesAsString

            # free memory
            #gc.collect()
        return number
    else:
        print("Stereo wave files are not supported yet")
        return None

def main():
    global models
    global modelTolerance
    global modelScore

    print("")
    print("This script can only process conf.CHUNK (currently: " + str(conf.CHUNK) +
          ") frames per loop so if the file contains a number of frames which is not divisible by conf.CHUNK the last few frames are dropped")

    model = []
    wavenumber = 1
    # TODO just for testing. recordsName as method argument needed
    fileName, modelName, minFrames, maxFrames, optimalFrames, iterations = "records/whistle", "", 3, 6, 4, 25#interactions.getTrainParameters()
    
    # do it while there are wave files
    
    while os.path.isfile(str(fileName) + "/" + str(wavenumber) + ".wav"):
        wf = wave.open(str(fileName) + "/" + str(wavenumber) + ".wav")
        number = preprocess(wf, fileName, wavenumber)
        model.append(copy.deepcopy(number))
        wf.close()
        wavenumber += 1

    if conf.ELIMINATE_BACKGROUND_NOISE:
        # TODO: bilde den durchschnitt der Hintergrundfrequenzen und ziehe diese von jeder aufnahme ab. Da fft sollten damit ja die Hintergrundfrequenzen ausgeloescht werden    
        print()

    print("Processed files, free up memory.")


    # the number of training files
    wavenumber -= 1
    # free memory
    gc.collect()

    #h = hpy()
    #print h.heap()


    if model != []:
        for j in model:
            tolerance = f.calculateTolerance(j, minFrames, maxFrames, optimalFrames, 0)
            diffrences = f.minimalizeFeaturesNew(j, optimalFrames, 0)
            maxDiffrence = diffrences[0]
            minDiffrence = diffrences[1]
            #for i in diffrences:
            #    if i > maxDiffrence:
            #        maxDiffrence = i
            #    if i < minDiffrence:
            #        minDiffrence = i
            #print("length diffrences: \t" + str(len(diffrences)))
            #print("maxDiffrence: \t" + str(maxDiffrence))
            print("minDiffrence: \t" + str(minDiffrence))
            print("tolerance: \t" + str(tolerance))

        

#TODO just for testing
main()