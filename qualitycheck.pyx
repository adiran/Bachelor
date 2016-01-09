import os.path
import functions as f
import wave
import config as conf
import numpy as np
cimport numpy as np
from scipy.fftpack import fft
import copy

# Cython stuff
CUINT64 = np.uint64
ctypedef np.uint64_t CUINT64_t
CINT16 = np.int16
ctypedef np.int16_t CINT16_t
CUINT32 = np.uint32
ctypedef np.uint32_t CUINT32_t

def qualityCheck(model, str fileName, int counter):

    #print("QS Number: " + str(counter) + " | fileName: " + fileName)

    cdef CUINT64_t tolerance = model.tolerance * conf.TOLERANCE_MULTIPLIER
    cdef list features = model.extractedFeatures
    cdef int wavenumber = 1
    cdef int result = 0
    cdef int modelPosition
    cdef bint first
    cdef int loops
    cdef bint switch
    cdef bytes framesAsString
    cdef np.ndarray[CINT16_t] frame
    cdef np.ndarray[CUINT32_t] feature
    cdef CUINT64_t compared

    # do it while there are wave files
    while os.path.isfile(str(fileName) + str(wavenumber) + ".wav"):
        wf = wave.open(str(fileName) + str(wavenumber) + ".wav")
        modelPosition = 0
        first = True

        # check wether the wave file is mono or stereo
        if wf.getnchannels() == 1:
            #print("File output" + str(wavenumber) + ".wav found. Processing it now...")
            loops = int(wf.getnframes() / conf.CHUNK)
            switch = True

#            print("Found " + str(wf.getnframes()) + " frames in this file.")
#            print("Run the loop " + str(loops) + " times, so the first " + str(loops * conf.CHUNK) + " frames will be processed and the last " + str(wf.getnframes() - (loops * conf.CHUNK)) + " dropped")
            for i in range(loops):

                framesAsString = wf.readframes(conf.CHUNK)
#                           print("Read " + str(len(frames)) + " frames.")

                if switch:
                    frame = np.fromstring(framesAsString, np.int16)
                    switch = False
                else:
                    switch = True
                    frame = np.append(
                        frame, np.fromstring(framesAsString, np.int16))
                    
                    feature = f.process(frame)
                    #print("Type: " + str(type(f.extractFeatures(feature)[0])))
                    feature = f.extractFeatures(feature)
                    # if we are not in the same frame of the model as last
                    # time we check if we are in the next frame
                    if f.compare(features[modelPosition], feature) > tolerance:
                        # print("length model: " + str(len(features)) + " | modelPosition + 1: " + str(modelPosition + 1))
                        # maby we are in the next model frame
                        if f.compare(features[modelPosition + 1], feature) < tolerance:
                            modelPosition += 1
#                            print("modelPosition: " + str(modelPosition))
#                            print("recognized the %d frame. frameCount = %d" %(modelPosition + 1, frameCount))
                            # recognized last frame of the features. Print it
                            # and reset featuresNumber and modelPosition
                            if modelPosition == (len(features) - 1):
                                result += 1
                                # print("fertig")
                                break
                        else:
                            first = True
        wavenumber += 1
        wf.close()
    return result
