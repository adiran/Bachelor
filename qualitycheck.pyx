"""Audio Trainer v1.0"""
# Imports of python libs
import os.path
import wave
import numpy as np
cimport numpy as np

# import of own scripts
import functions as f
import config as conf

# Cython stuff
CUINT64 = np.float64
ctypedef np.float64_t CUINT64_t
CINT16 = np.int16
ctypedef np.int16_t CINT16_t

cpdef void qualityCheck(tuple data) except *:
    cdef model = data[0]
    cdef str fileName = data[1]
    cdef int counter = data[2]
    cdef np.ndarray[CUINT64_t] threshold = model.threshold
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
    cdef cepstrum = conf.CEPSTRUM


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
                    
                    if cepstrum:
                        feature = f.processCepstrum(frame)
                    else:
                        feature = f.processSpectrum(frame)
                    compared = f.compare(feature, features[modelPosition])
                    if compared < threshold[modelPosition]:
                        recognized = True
                        frameCount = conf.FRAME_COUNT
                    # if we are not in the same frame of the model as last
                    # time we check if we are in the next frame
                    elif recognized:
                        # maby we are in the next model frame
                        if f.compare(feature, features[modelPosition + 1]) < threshold[modelPosition + 1]:
                            recognized = True
                            frameCount = conf.FRAME_COUNT
                            modelPosition += 1
                            # recognized last frame of the features. Count it
                            # and reset featuresNumber and modelPosition
                            if modelPosition == (len(features) - 1):
                                result += 1
                                break
                        else:
                            if frameCount > 0:
                                frameCount -= 1
                            else:
                                recognized = False
                                modelPosition = 0
        wavenumber += 1
        wf.close()
    model.matches = result
    model.calculateScore()
    f.storeModel(model, True, counter)

