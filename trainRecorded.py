"""Audio Trainer v1.0"""
# Imports of python libs
import wave
import time
import os.path
import numpy as np
import multiprocessing
import copy

# import of own scripts
import functions as f
import qualitycheck as q
import interactions
import model as modelImport
import config as conf


def preprocess(in_data):
    wf = in_data[0]
    wavenumber = in_data[1]
    # check wether the wave file is mono or stereo
    if wf.getnchannels() == 1:
        loops = int(wf.getnframes() / conf.CHUNK)
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
                if conf.CEPSTRUM:
                    if conf.LIFTERING:
                        frame = f.processCepsLiftering(frame)
                    else:
                        frame = f.processCepstrum(frame)
                else:
                    frame = f.processSpectrum(frame)
                number.append(frame)
                switch = True
            del framesAsString

        print("Processed file " + str(wavenumber) + ".wav")
        return number
    else:
        print("Stereo wave files are not supported yet")
        return None

def main():
    global models
    global modelThreshold
    global modelScore

    print("")
    print("This script can only process conf.CHUNK (currently: " + str(conf.CHUNK) +
          ") frames per loop so if the file contains a number of frames which is not divisible by conf.CHUNK the last few frames are dropped")

    model = []
    wavenumber = 1
    fileName, modelName, optimalFrames, scriptpath = interactions.getTrainParameters()
    beginning = time.time()
    # do it while there are wave files
    wf = []
    while os.path.isfile(str(fileName) + "/" + str(wavenumber) + ".wav"):
        wf.append((wave.open(str(fileName) + "/" + str(wavenumber) + ".wav"), wavenumber))
        print("File " + str(fileName) + "/" + str(wavenumber) +
                ".wav found.")
        wavenumber += 1
    for i in wf:
        model.append(preprocess(i))
    for i in wf:
        i[0].close()
    wavenumber -= 1
    print("Processed " + str(wavenumber) + " files in " + str(time.time() - beginning) + " seconds, minimalize them.")

    if model != []:
        data = []
        for i in range(wavenumber):
            data.append(
                (model[i],
                 optimalFrames))
        beginning = time.time()
        f.clearTmpFolder()
        pool = multiprocessing.Pool(processes=4)
        result = pool.map(f.minimalizeAndCalcThreshold, data)
        minimalizedRecords = []
        calculatedThresholds = []
        for i in result:
            minimalizedRecords.append(i[0])
            calculatedThresholds.append(i[1])

        zeroFrame = np.zeros(conf.FEATURES_PER_FRAME, dtype=np.float64)
        models = []
        for i in range(len(minimalizedRecords)):
            features = copy.deepcopy(minimalizedRecords[i])
            tmpFeatures = [copy.deepcopy(zeroFrame) for number in range(optimalFrames)]
            tmpCounter = [0 for number in range(optimalFrames)]
            counter = 0.
            posCounter = [0 for number in range(len(minimalizedRecords))]
            # for every frame in this record try if we find mergable frames
            for h in range(optimalFrames):    
                # we try all recordings
                for j in range(len(minimalizedRecords)):
                    if f.compare(minimalizedRecords[j][h], features[h]) < calculatedThresholds[i][h]:
                        tmpFeatures[h] += minimalizedRecords[j][h]
                        tmpCounter[h] += 1
            for h in range(optimalFrames):
                tmpFeatures[h] = np.divide(tmpFeatures[h], tmpCounter[h])
                counter += tmpCounter[h]
            counter /= optimalFrames
            models.append(modelImport.Model(tmpFeatures, calculatedThresholds[i], modelName, tmpCounter, scriptpath))


        print()
        print("Computed the models in " + str(time.time() - beginning) + " seconds. Compute their score.")
        print()
        beginning = time.time()
        data = []
        for i in range(len(models)):
            data.append((models[i], fileName, i))
        pool = multiprocessing.Pool(processes=4)
        pool.map(q.qualityCheck, data)
        models = f.loadModels(tmp=True)
        print("Computed the scores in " + str(time.time() - beginning) + " seconds.")
        print()
        for i in range(len(models)):
            print("Model Nr:\t" +
                  str(i +
                      1) +
                  " | Frames:\t" +
                  str(len(models[i].features)) +
                  " | Matches:\t" +
                  str(models[i].matches) +
                  " | Influenced by:\t" +
                  str(models[i].influencedBy) +
                  " | Threshold:\t" +
                  str(models[i].threshold) +
                  " | Score:\t" +
                  str(models[i].score))

        # get the model number and substract 1 because list indexing starts
        # with 0
        modelNumber = interactions.getModelNumber(len(models)+1) - 1
        print("You selected Model " + str(modelNumber) + " with " + str(models[modelNumber].matches) + " Matches and a Score of: " + str(models[modelNumber].score))
        f.storeModel(models[modelNumber])