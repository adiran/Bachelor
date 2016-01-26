import numpy as np
cimport numpy as np
from scipy.fftpack import rfft


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
    global modelNumber
    global frameCount
    global number
    global switch
    global frame

    # load the stored models
    models = f.loadModels()
    
    # stores the position in the model
    modelPosition = 0

    # stores the model which we assume to be recognized. -1 if no model has
    # been recognized
    modelNumber = -1

    # decrements with every frame that is not recognized as an model frame. If
    # zero we don't have a match
    frameCount = conf.FRAME_COUNT

    # preprocessed audio data
    number = []

    # we store 2 frames of collected audio data as 1 frame in sample so here
    # is a switch
    switch = True

    # used for storing half of a frame because of the switch
    frame = []    


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
        global modelNumber
        global frameCount
        global models
#        print("frame = np.append(frame, np.fromstring(in_data, np.int16))")
        frame = np.append(frame, np.fromstring(in_data, np.int16))
#        print("data = np.abs(np.split(fft(frame), 2))")
        data = f.process(frame)
#        print("number.append(f.extractFeatures(data[0]))")
        # if modelNumber < 0 then we don't recognized a model in the frame
        # before
        if modelNumber < 0:
            #print("we are in the first check")
            for i in range(len(models)):
                #                print("testing model %d" %(i))
                # if f.compare is < conf.TOLERANCE then we have a match and set
                # modelNumber
                compared = f.compare(models[i].features[0], data)
                print(str(compared) + " - " + str(models[i].tolerance) + " = " + str(compared - models[i].tolerance))
                if compared < models[i].tolerance:
                    print("recognized the first frame")
                    modelNumber = i
                    frameCount = conf.FRAME_COUNT
                    break
        # we had a match in the last frame
        else:
            global modelPosition
            # if we are not in the same frame of the model as last time we
            # check if we are in the next frame
            if f.compare(models[modelNumber].features[modelPosition], data) > models[modelNumber].tolerance:
                frameCount = conf.FRAME_COUNT
            else:
                # maby we are in the next model frame
                if f.compare(models[modelNumber].features[modelPosition + 1], data) < models[modelNumber].tolerance:
                    modelPosition += 1
                    frameCount = conf.FRAME_COUNT
                    print("recognized the %d frame. frameCount = %d" %
                          (modelPosition + 1, frameCount))
                    # recognized last frame of the model. Print it and reset
                    # modelNumber and modelPosition
                    if modelPosition == (len(models[modelNumber].features) - 1):
                        print("Recognized model number %d" % (modelNumber))
                        modelNumber = -1
                        modelPosition = 0
                # we have not recognized a frame. reset modelNumber and
                # modelPosition
                else:
                    # we test a few more frames (as set in config.py) before we
                    # reset
                    if frameCount > 0:
                        frameCount -= 1
                        print("Framecount: " + str(frameCount))
                    else:
                        modelNumber = -1
                        modelPosition = 0
        switch = True