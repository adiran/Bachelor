import numpy as np
cimport numpy as np
from scipy.fftpack import rfft
import time as timebib


#own imports
import config as conf
import model
import functions as f


# Cython stuff
CFLOAT = np.float64
ctypedef np.float64_t CFLOAT_t
CUINT64 = np.uint64
ctypedef np.uint64_t CUINT64_t
CINT16 = np.int16
ctypedef np.int16_t CINT16_t

cpdef setup():
    global models
    global modelPosition
    global frameCount
    global number
    global switch
    global frame

    # load the stored models
    models = f.loadActivatedModels()

    for i in range(len(models)):
        print("Model Nr:\t" +
             str(i +
                 1) +
             " | Name:\t" +
             str(models[i].name) +
             " | Frames:\t" +
             str(len(models[i].features)) +
             " | Matches:\t" +
             str(models[i].matches) +
             " | Influenced by:\t" +
             str(models[i].influencedBy) +
             " | Tolerance:\t" +
             str(models[i].tolerance) +
             " | Score:\t" +
             str(models[i].score) +
             " | Loaded:\t" +
             str(models[i].loaded))
    
    # stores the position in the model
    modelPosition = np.zeros_like(models)

    # decrements with every frame that is not recognized as an model frame. If
    # zero we don't have a match
    frameCount = np.zeros_like(models)

    # preprocessed audio data
    number = []

    # we store 2 frames of collected audio data as 1 frame in sample so here
    # is a switch
    switch = True

    # used for storing half of a frame because of the switch
    frame = []


cpdef beginning():
    global beginning

    # TODO just for testing
    beginning = timebib.time()

cpdef listen(in_data):
    cdef np.ndarray[CUINT64_t] data

    global switch
    global frame

    # capturing the first half of the frame
    if switch:
        #        print("frame = np.fromstring(in_data, np.int16)")
        frame = np.fromstring(in_data, np.int16)
        switch = False

    # capturing the second half and preprocessing
    else:
        global modelPosition
        global frameCount
        global models
#        print("frame = np.append(frame, np.fromstring(in_data, np.int16))")
        frame = np.append(frame, np.fromstring(in_data, np.int16))
#        print("data = np.abs(np.split(fft(frame), 2))")
        data = f.process(frame)
#        print("number.append(f.extractFeatures(data[0]))")
        for i in range(models.size):
            if f.compare(models[i].features[modelPosition[i]], data) < models[i].tolerance:
                frameCount[i] = conf.FRAME_COUNT
            elif f.compare(models[i].features[modelPosition[i] + 1], data) < models[i].tolerance:
                modelPosition[i] += 1
                print("Recognized " + str(modelPosition[i]) + ". frame.")
                frameCount[i] = conf.FRAME_COUNT
                if modelPosition[i] == (len(models[i].features) - 1):
                    global beginning
                    recognizeTime = int(timebib.time() - beginning)
                    recognizeTimeMinutes = int(recognizeTime / 60)
                    recognizeTime -= recognizeTimeMinutes * 60
                    print("Recognized model " + models[i].name + " after " + str(recognizeTimeMinutes) + ":" + str(recognizeTime) + ".")
                    modelPosition = np.zeros_like(models)
                    frameCount = np.zeros_like(models)
                    break
            else:
                # we test a few more frames (as set in config.py) before we
                # reset
                if frameCount[i] > 0:
                    frameCount[i] -= 1
                    #print("Framecount: " + str(frameCount))
                else:
                    modelPosition[i] = 0
        switch = True