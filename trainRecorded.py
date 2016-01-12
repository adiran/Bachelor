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
from scipy.fftpack import fft

# import of own scripts
import functions as f
import qualitycheck as q
import interactions



def modelMergeNew(data):
    model, minFrames, maxFrames, name, iteration = data
    f.modelMergeNew(model, minFrames, maxFrames, name, iteration)

def main():
    global models
    global modelTolerance
    global modelScore

    print("This script can only process conf.CHUNK (currently: " + str(conf.CHUNK) +
          ") frames per loop so if the file contains a number of frames which is not divisible by conf.CHUNK the last few frames are dropped")

    model = []
    wavenumber = 1
    # TODO just for testing. recordsName as method argument needed
    fileName, modelName, minFrames, maxFrames, iterations = interactions.getTrainParameters()
    
    # do it while there are wave files
    while os.path.isfile(str(fileName) + "/" + str(wavenumber) + ".wav"):
        wf = wave.open(str(fileName) + "/" + str(wavenumber) + ".wav")

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
#                print("AudioLevel of these frames: " + str(audioLevel))
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
                        number.append(f.process(frame))
                        switch = True

            model.append(number)
#            print("Length model: " + str(len(model)))
            wf.close()
            wavenumber += 1

        else:
            print("Stereo wave files are not supported yet")

    # the number of training files
    wavenumber -= 1
    

    #print("Type Model: " + str(type(model)) + " | Type Model[0]: " + str(type(model[0])) + " | Type Model[0][0]: " + str(type(model[0][0])) + " | Type Model[0][0][0]: " + str(type(model[0][0][0])))

    if model != []:
        oldModelScore = -1
        data = []
        minFramesList = []
        maxFramesList = []
        for i in range(iterations):
            pos1 = model[0]
            model[0] = model[i]
            model[i] = pos1
            #print("Pack data that beginns with: " + str(model[0][0][0]))
            #print("Next record starts with: " + str(model[1][0]))
            data.append(
                (copy.deepcopy(model),
                 minFrames,
                 maxFrames,
                 modelName,
                 i))

        # for i in data:
         #   model, minFrames, maxFrames, name, iteration = i
          #  print("Model beginns with: " + str(model[0][0][0]))

        print()
        print("Compute " + str(iterations) +
              " diffrent models.")
        beginning = time.time()
        f.clearTmpFolder()
        multipro = True
        if multipro:
            pool = multiprocessing.Pool(processes=4)
            pool.map(f.modelMergeNew, data)
        else:
            for i in data:
                f.modelMergeNew(i)

#        result.get()

        models = f.loadModels(tmp=True)
        print("Computed " + str(len(models)) +
              " models in " + str(time.time() - beginning) + "... Compute their score.")

        for i in range(len(models)):
            #print("Length extractedFeatures: " + str(len(models[i].extractedFeatures[0])))
            models[i].score = q.qualityCheck(models[i], fileName, i)
            #print("Type Features: " + str(type(models[i].features[0])) + " | Type FeatureValue: " + str(type(models[i].features[0][0])))
            print("Model Nr: " +
                  str(i +
                      1) +
                  " | Frames: " +
                  str(len(models[i].features)) +
                  " | Score: " +
                  str(models[i].score) +
                  " | Tolerance: " +
                  str(models[i].tolerance))

        # get the model number and substract 1 because list indexing starts
        # with 0
        modelNumber = interactions.getModelNumber(len(models)+1) - 1
        f.storeModel(models[modelNumber])
