"""Audio Trainer v1.0"""
# Imports of python libs
import numpy as np
cimport numpy as np
from subprocess import call

# import of own scripts
import config as conf
import functions as f


# Cython stuff
CFLOAT = np.float64
ctypedef np.float64_t CFLOAT_t
CUINT64 = np.float64
ctypedef np.float64_t CUINT64_t
CINT16 = np.int16
ctypedef np.int16_t CINT16_t

cpdef setup():
    global models
    global modelPosition
    global frameCount
    global number
    global switch
    global frame
    global recognition
    global cepstrum

    # counter of frames that should be ignored after recognition
    recognition = 0

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
             " | Threshold:\t" +
             str(models[i].threshold) +
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

    # wether Cepstrum or Spectrum should be used for Feature Extraction
    cepstrum = conf.CEPSTRUM


cpdef listen(in_data):
    cdef np.ndarray[CUINT64_t] data
    global switch
    global frame
    global recognition

    if recognition == 0:
        # capturing the first half of the frame
        if switch:
            frame = np.fromstring(in_data, np.int16)
            switch = False

        # capturing the second half and preprocessing
        else:
            global modelPosition
            global frameCount
            global models
            frame = np.append(frame, np.fromstring(in_data, np.int16))
            if cepstrum:
                data = f.processCepstrum(frame)
            else:
                data = f.processSpectrum(frame)
            for i in range(models.size):
                # check if we are in the same model frame as before
                if f.compare(data, models[i].features[modelPosition[i]]) < models[i].threshold[modelPosition[i]]:
                    frameCount[i] = conf.FRAME_COUNT
                # if not maby we are in the next frame
                elif f.compare(data, models[i].features[modelPosition[i] + 1]) < models[i].threshold[modelPosition[i]]:
                    modelPosition[i] += 1
                    frameCount[i] = conf.FRAME_COUNT
                    if modelPosition[i] == (len(models[i].features) - 1):
                        call(["python2.7", str(models[i].script), str(models[i].name)])
                        modelPosition = np.zeros_like(models)
                        frameCount = np.zeros_like(models)
                        recognition = conf.SKIP_AFTER_RECOGNITION
                        break
                # we don't recognized the current or the next frame
                else:
                    # we test a few more frames (as set in config.py) before we
                    # reset
                    if frameCount[i] > 0:
                        frameCount[i] -= 1
                    else:
                        modelPosition[i] = 0
            switch = True
    else:
        recognition -= 1