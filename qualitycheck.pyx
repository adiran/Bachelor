import os.path
import functions as f
import wave
import config as conf
import numpy as np
cimport numpy as np
from scipy.fftpack import fft
import copy

# Cython stuff
CUINT64 = np.float64
ctypedef np.float64_t CUINT64_t
CINT16 = np.int16
ctypedef np.int16_t CINT16_t

def qualityCheck(tuple data):
    cdef model = data[0]
    cdef str fileName = data[1]
    cdef int counter = data[2]

    print("QS Number: " + str(counter) + " | fileName: " + fileName)

    cdef CUINT64_t tolerance = model.tolerance * conf.TOLERANCE_MULTIPLIER
    cdef list features = model.features
    cdef int wavenumber = 1
    cdef int result = 0
    cdef int modelPosition
    cdef int loops
    cdef int frameCount
    cdef bint switch
    cdef bint recognized
    cdef bytes framesAsString
    cdef np.ndarray[CINT16_t] frame
    cdef np.ndarray[CUINT64_t] feature
    cdef CUINT64_t compared

    # do it while there are wave files
    while os.path.isfile(str(fileName) + "/" + str(wavenumber) + ".wav"):
        wf = wave.open(str(fileName) + "/" + str(wavenumber) + ".wav")
        modelPosition = 0
        recognized = False
        frameCount = conf.FRAME_COUNT

        # check wether the wave file is mono or stereo
        if wf.getnchannels() == 1:
        
            loops = int(wf.getnframes() / conf.CHUNK)
            switch = True
            for i in range(loops):

                framesAsString = wf.readframes(conf.CHUNK)
                if switch:
                    frame = np.fromstring(framesAsString, np.int16)
                    switch = False
                else:
                    switch = True
                    frame = np.append(
                        frame, np.fromstring(framesAsString, np.int16))
                    
                    feature = f.process(frame)
                    compared = f.compare(features[modelPosition], feature)
                   # print("QS Number: " + str(counter) + ", frame Nr: " + str((i+1)/2) + ", compared: " + str(compared) + ", tolerance-compared: " + str(tolerance - compared))
                    if compared < tolerance:
                        recognized = True
                        frameCount = conf.FRAME_COUNT
                        #print("QS Number: " + str(counter) + ", recognized frame " + str(modelPosition))
                    # if we are not in the same frame of the model as last
                    # time we check if we are in the next frame
                    elif recognized:
                        # print("length model: " + str(len(features)) + " | modelPosition + 1: " + str(modelPosition + 1))
                        # maby we are in the next model frame
                        if f.compare(features[modelPosition + 1], feature) < tolerance:
                            recognized = True
                            frameCount = conf.FRAME_COUNT
                            modelPosition += 1
                            #print("QS Number: " + str(counter) + ", recognized frame " + str(modelPosition))
                            # recognized last frame of the features. Print it
                            # and reset featuresNumber and modelPosition
                            if modelPosition == (len(features) - 1):
                                result += 1
                                #print("fertig")
                                break
                        else:
                            #print("QS Number: " + str(counter) + ", frameCount: " + str(frameCount))
                            if frameCount > 0:
                                frameCount -= 1
                            else:
                                recognized = False
        wavenumber += 1
        wf.close()
    model.matches = result
    model.calculateScore()
    print("QS Number: " + str(counter) + " | Matches: " + str(model.matches))
    f.storeModel(model, True, counter)

