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
import model as modelImport



def modelMergeNew(data):
    model, minFrames, maxFrames, name, iteration = data
    f.modelMergeNew(model, minFrames, maxFrames, name, iteration)

#@profile
def preprocess(in_data):
    wf = in_data[0]
    wavenumber = in_data[1]
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
        print("Processed file " + str(wavenumber) + ".wav")
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
    beginning = time.time()
    # do it while there are wave files
    wf = []
    while os.path.isfile(str(fileName) + "/" + str(wavenumber) + ".wav"):
        wf.append((wave.open(str(fileName) + "/" + str(wavenumber) + ".wav"), wavenumber))
        print("File " + str(fileName) + "/" + str(wavenumber) +
                ".wav found.")
        wavenumber += 1
    multipro = False
    if multipro:
        pool = multiprocessing.Pool(processes=4)
        model = pool.map(preprocess, wf)
    else:
        for i in wf:
            model.append(preprocess(i))
    for i in wf:
        i[0].close()
    if conf.ELIMINATE_BACKGROUND_NOISE:
        # TODO: bilde den durchschnitt der Hintergrundfrequenzen und ziehe diese von jeder aufnahme ab. Da fft sollten damit ja die Hintergrundfrequenzen ausgeloescht werden    
        print()
    wavenumber -= 1
    print("Processed " + str(wavenumber) + " files in " + str(time.time() - beginning) + " seconds, minimalize them.")

    if model != []:
        data = []
        for i in range(wavenumber):
            data.append(
                (model[i],
                 optimalFrames,
                 i))
        beginning = time.time()
        f.clearTmpFolder()
        #TODO just for testing drop that iout before finish
        multipro = True
        print(str(len(data)))
        if multipro:
            pool = multiprocessing.Pool(processes=4)
            dasReturnZeugs = pool.map(f.minimalizeAndCalcTolerance, data)
        else:
            for i in data:
                f.modelMergeNew(i)
        minimalizedRecords = []
        calculatedTolerances = []
        for i in dasReturnZeugs:
            minimalizedRecords.append(i[0])
            calculatedTolerances.append(i[1])

        zeroFrame = np.zeros(conf.FEATURES_PER_FRAME, dtype=np.float64)
        models = []
        for i in range(len(minimalizedRecords)):
            features = copy.deepcopy(minimalizedRecords[i])
            tmpFeatures = [copy.deepcopy(zeroFrame) for number in range(optimalFrames)]
            tmpCounter = [0 for number in range(optimalFrames)]
            for j in range(len(minimalizedRecords)):
                for h in range(optimalFrames):
                    if f.compare(minimalizedRecords[j][h], features[h]) < calculatedTolerances[i][h]:
                        tmpFeatures[h] += minimalizedRecords[j][h]
                        tmpCounter[h] += 1
            for h in range(optimalFrames):
                tmpFeatures[h] = np.divide(tmpFeatures[h], tmpCounter[h])
            models.append(modelImport.Model(tmpFeatures, calculatedTolerances[i], modelName, 24))


        print()
        print("Computed the models in " + str(time.time() - beginning) + " seconds. Compute their score.")
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