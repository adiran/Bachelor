"""Audio Trainer v1.0"""
# Imports of python libs
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
def preprocess(wf):
    # check wether the wave file is mono or stereo
    if wf.getnchannels() == 1:
        loops = int(wf.getnframes() / conf.CHUNK)
        framesAfterSound = conf.FRAMES_AFTER_SOUND
        number = []
        switch = True
        for i in range(loops):
            
            framesAsString = wf.readframes(512)
            if switch:
                frame = np.fromstring(framesAsString, np.int16)
                switch = False
            else:
                frame = np.append(
                    frame, np.fromstring(framesAsString, np.int16))
                #print("1 Length frame: " + str(len(frame)) + " | frame[0]: " + str(frame[0]))
                frame = f.process(frame)
                #print("2 Length frame: " + str(len(frame)) + " | frame[0]: " + str(frame[0]))
                number.append(frame)
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
    fileName, modelName, optimalFrames, iterations = interactions.getTrainParameters()
    
    # do it while there are wave files
    
    while os.path.isfile(str(fileName) + "/" + str(wavenumber) + ".wav"):
        wf = wave.open(str(fileName) + "/" + str(wavenumber) + ".wav")
        print("File " + str(fileName) + "/" + str(wavenumber) +
                ".wav found. Processing it now...")
        number = preprocess(wf)
        #print("Number[0][0]: " + str(number[0][0]))
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

    print("Type Model: " + str(type(model)))
    print("Length Model: " + str(len(model)))
    print("Type Model[0]: " + str(type(model[0])))
    print("Length Model[0]: " + str(len(model[0])))
    print("Type Model[0][0]: " + str(type(model[0][0])))
    print("Length Model[0][0]: " + str(len(model[0][0])))
    print("Type Model[0][0][0]: " + str(type(model[0][0][0])))
    #print("Length Model[0][0][0]: " + str(len(model[0][0][0])))

    print()
    print("Compute " + str(iterations) +
          " diffrent models.")

    if model != []:
        oldModelScore = -1
        data = []
        minFramesList = []
        maxFramesList = []
        for i in range(iterations):
            # free memory
            #gc.collect()
            print(i)
            pos1 = copy.deepcopy(model[0])
            model[0] = copy.deepcopy(model[i])
            model[i] = pos1
            #print("Pack data that beginns with: " + str(model[0][0][0]))
            #print("Next record starts with: " + str(model[1][0]))
            data.append(
                (copy.deepcopy(model),
                 optimalFrames,
                 modelName,
                 i))

        print("Length model: " + str(len(model)) + " | model[0]: " + str(len(model[0])) + " | model[0][0]: " + str(len(model[0][0])) + " | model[0][0][0]: " + str((model[0][0][0])))
        beginning = time.time()
        f.clearTmpFolder()
        #TODO just for testing drop that iout before finish
        multipro = True
        if multipro:
            pool = multiprocessing.Pool(processes=4)
            pool.map(f.modelMergeNew, data)
        else:
            for i in data:
                f.modelMergeNew(i)

        print()
        print("Computed the models in " + str(time.time() - beginning) + " seconds. Load them for scoring. Don't worry this might take some time...")
        print()
        beginning = time.time()
        models = f.loadModels(tmp=True)
        print("Loaded " + str(len(models)) +
              " models in " + str(time.time() - beginning) + " seconds. Compute their score.")
        print()
        beginning = time.time()
        data = []
        for i in range(len(models)):
            data.append((models[i], fileName, i))
        multipro = True
        if multipro:
            pool = multiprocessing.Pool(processes=4)
            pool.map(q.qualityCheck, data)
        else:
            for i in data:
                q.qualityCheck(i)
        models = f.loadModels(tmp=True)
        print("Computed the scores in " + str(time.time() - beginning) + " seconds.")
        print()
        for i in range(len(models)):
            #print("Length extractedFeatures: " + str(len(models[i].extractedFeatures[0])))
            #print("Type Features: " + str(type(models[i].features[0])) + " | Type FeatureValue: " + str(type(models[i].features[0][0])))
            print("Model Nr:\t" +
                  str(i +
                      1) +
                  " | Frames:\t" +
                  str(len(models[i].features)) +
                  " | Matches:\t" +
                  str(models[i].matches) +
                  " | Influenced by:\t" +
                  str(models[i].influencedBy) +
                  " | Tolerance:\t" +
                  str(models[i].tolerance) +
                  " | Score:\t" +
                  str(models[i].score))

        # get the model number and substract 1 because list indexing starts
        # with 0
        modelNumber = interactions.getModelNumber(len(models)+1) - 1
        print("You selected Model " + str(modelNumber) + " with " + str(models[modelNumber].matches) + " Matches and a Score of: " + str(models[modelNumber].score))
        f.storeModel(models[modelNumber])


#TODO just for testing
#main()