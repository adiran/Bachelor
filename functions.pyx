import numpy as np
cimport numpy as np
from scipy.fftpack import fft, ifft
import copy
import os
import pickle
import time
import random
#from features import mfcc
import math

#own imports
import config as conf
import trainRecorded
import model
import interactions

#TODO just for testing
import sys

# Cython stuff
CFLOAT = np.float64
ctypedef np.float64_t CFLOAT_
tCUINT64 = np.float64
ctypedef np.float64_t CUINT64_t
CINT16 = np.int16
ctypedef np.int16_t CINT16_t


# summe der quadrate der distanzen zweier int arrays


cpdef CUINT64_t compare(np.ndarray[CUINT64_t] recordData, np.ndarray[CUINT64_t] modelData):
    cdef CUINT64_t result = 0
    cdef Py_ssize_t i
    cdef CUINT64_t tmp1
    cdef CUINT64_t tmp2
    if len(recordData) == len(modelData):
        for i in range(len(recordData)):
            tmp2 = <CUINT64_t>abs(modelData[i])
            if math.isnan(tmp2) == False:
                tmp1 = <CUINT64_t>abs(recordData[i])
                if tmp1 > tmp2:
                    result += tmp1 - tmp2
                else:
                    result += tmp2 - tmp1
    else:
        print("Compare failed because of different shapes")
        result = 18446744073709551615
    return result


cpdef tuple minimalizeAndCalcTolerance(tuple in_data):
    cdef list in_array = copy.deepcopy(in_data[0])
    cdef int optimalFrames = in_data[1]
    cdef int iteration = in_data[2]
    cdef list frameMember = []
    cdef list diffrences = []
    cdef CUINT64_t diffrence = 0.
    cdef CUINT64_t compared = 0.
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t pos = 0
    cdef int posBeg = 0
    cdef int posEnd = 0
    cdef int posBegBefore = 0
    cdef int posEndBefore = 0
    cdef int posBegAfter = 0
    cdef int posEndAfter = 0
    cdef int counter = 0
    cdef int counterBefore = 0
    cdef int counterAfter = 0
    cdef int tmpFrameMember = 0
    cdef int tmpFrameMemberBeforeAfter = 0
    cdef np.ndarray[CUINT64_t] tmpFrame = np.zeros_like(in_array[0], dtype=np.float64)
    cdef np.ndarray[CUINT64_t] tmpFrameBefore = np.zeros_like(in_array[0], dtype=np.float64)
    cdef np.ndarray[CUINT64_t] tmpFrameAfter = np.zeros_like(in_array[0], dtype=np.float64)
    cdef CUINT64_t maxDiffrence = 0
    cdef int length = len(in_array)
    cdef np.ndarray[int] counterFrames = np.zeros(optimalFrames, dtype=int)
    cdef list result = []
    cdef np.ndarray[CUINT64_t] resultTolerance = np.zeros(optimalFrames, dtype=np.float64)
    for i in range(length):
        frameMember.append(i)
        if i < length -1:
            diffrences.append(compare(in_array[i], in_array[i+1]))
    while length > optimalFrames:
        diffrence = np.finfo(np.float64).max - np.finfo(np.float64).eps
        counter = 0
        counterAfter = 0
        posBeg = -1
        posEnd = -1
        posBegBefore = -1
        posEndBefore = -1
        posBegAfter = -1
        posEndAfter = -1
        pos = -1
        tmpFrameMemberBeforeAfter = -1
        tmpFrame = np.zeros_like(in_array[0], dtype=np.float64)
        tmpFrameBefore = np.zeros_like(in_array[0], dtype=np.float64)
        tmpFrameAfter = np.zeros_like(in_array[0], dtype=np.float64)
        for i in range(len(diffrences)):
            if diffrences[i] < diffrence:
                pos = i
                diffrence = diffrences[i]
        if diffrence > maxDiffrence:
            maxDiffrence = diffrence
        tmpFrameMember = frameMember[pos + 1]
        j = pos + 1
        while frameMember[j] == tmpFrameMember:
            frameMember[j] = frameMember[pos]
            if j + 1 < len(frameMember):
                j += 1
            else:
                break
        tmpFrameMember = frameMember[pos]
        diffrences[pos] = np.finfo(np.float64).max
        #print("Set diffrences[" + str(pos) + "] to " + str(diffrences[pos]))
        for i in range(len(in_array)):
            if frameMember[i] < tmpFrameMember:
                if tmpFrameMemberBeforeAfter == frameMember[i]:
                    posEndBefore = i
                    counterBefore += 1
                else:
                    tmpFrameMemberBeforeAfter = frameMember[i]
                    posBegBefore = i
                    posEndBefore = i
                    counterBefore = 1
            elif frameMember[i] == tmpFrameMember:
                if posBeg == -1:
                    posBeg = i
                else:
                    posEnd = i
                    if (i + 1) < len(in_array):
                        if frameMember[i + 1] > tmpFrameMember:
                            tmpFrameMemberBeforeAfter = frameMember[i + 1]
                            posBegAfter = i + 1
                counter += 1
            elif frameMember[i] == tmpFrameMemberBeforeAfter:
                counterAfter += 1
                posEndAfter = i
            # we got the frame before the merge, the frame to merge and the frame after the merge. we don't need to go further through the array
            else:
                break
        for i in range(posBeg, posEnd + 1):
            tmpFrame = np.add(tmpFrame, np.divide(in_array[i], np.float64(counter)))
        #print("posBeg: " + str(posBeg) + " posEnd: " + str(posEnd) + " posBegBefore: " + str(posBegBefore) + " posEndBefore: " + str(posEndBefore) + " posBegAfter: " + str(posBegAfter) + " posEndAfter: " + str(posEndAfter))
        # if there are frames befor the merged frames 
        if posBegBefore > -1:
            for i in range(posBegBefore, posEndBefore + 1):
                tmpFrameBefore = np.add(tmpFrameBefore, np.divide(in_array[i], np.float64(counterBefore)))
            diffrences[posEndBefore] = compare(tmpFrameBefore, tmpFrame)
            #print("Recalculated diffrences[" + str(posEndBefore) + "] to " +str(diffrences[posEndBefore]))
        # if there are frames after the merged frames
        if posEnd < len(diffrences):
            for i in range(posBegAfter, posEndAfter + 1):
                tmpFrameAfter = np.add(tmpFrameAfter, np.divide(in_array[i], np.float64(counterAfter)))
            diffrences[posEnd] = compare(tmpFrame, tmpFrameAfter)
            #print("Recalculated diffrences[" + str(posEnd) + "] to " +str(diffrences[posEnd]))
        length -= 1
    tmpFrameMember = 0
    j = 0
    for i in range(len(in_array)):
        if tmpFrameMember == frameMember[i]:
            counterFrames[j] += 1
        else:
            tmpFrameMember = frameMember[i]
            j += 1
            counterFrames[j] += 1
    j = 0
    tmpFrame = np.zeros_like(in_array[0], dtype=np.float64)
    counter = counterFrames[0]
    for i in range(len(in_array)):
        if counter > 0:
            tmpFrame = np.add(tmpFrame, np.divide(in_array[i], np.float64(counterFrames[j])))
            counter -= 1
        else:
            j += 1
            result.append(tmpFrame)
            tmpFrame = np.divide(in_array[i], np.float64(counterFrames[j]))
            counter = counterFrames[j] - 1
    result.append(tmpFrame)
    if conf.FREQUENCY_BAND_TRAINING:
        #TODO
        print("Not Yet implementatet")
    j = 0
    for i in range(len(in_array) - 1):
        if frameMember[i] == frameMember[i + 1]:
            compared = compare(in_array[i], in_array[i + 1])
            if compared > resultTolerance[j]:
                resultTolerance[j] = compared
        else:
            j += 1
            if i < len(in_array) - 3:
                if frameMember[i + 1] != frameMember[i + 2]:
                    resultTolerance[j] = maxDiffrence
    if resultTolerance[optimalFrames - 1] == 0.:
        resultTolerance[optimalFrames - 1] = maxDiffrence
    for i in range(optimalFrames):
        resultTolerance[i] = (resultTolerance[i] + maxDiffrence)/2.
    return (result, resultTolerance)


# loads previously stored models
cpdef np.ndarray loadModels(bint tmp = False):
    cdef list result =  []
    cdef str directory
    if tmp:
        directory = conf.TMP_DIR
    else:
        directory = conf.MODELS_DIR
    for file in os.listdir(directory):
        if file != ".keep":
            openFile = open((directory + "/" + file), "rb")
            result.append(pickle.load(openFile))
            openFile.close()
    return np.asarray(result)

# loads all previously stored models that are marked as active
cpdef np.ndarray loadActivatedModels():
    cdef np.ndarray models = loadModels()
    cdef list result = []
    for i in models:
        if i.loaded:
            result.append(i)
    return np.asarray(result)



# stores a model at the modelFolder or in tmp folder
cpdef void storeModel(model, bint tmp=False, int iteration=0) except *:
    cdef str fileName
    cdef str dirName
    
    if tmp:
        fileName = model.name + str(iteration)
        dirName = conf.TMP_DIR + "/"
    else:
        dirName = conf.MODELS_DIR + "/"
        fileName = model.name
        while os.path.isfile(str(dirName) + str(fileName)):
            fileName = interactions.getDifferentModelFileName(fileName)
            model.name = fileName

    outputFile = open(dirName + fileName, "wb")
    pickle.dump(model, outputFile, pickle.HIGHEST_PROTOCOL)
    outputFile.close()

# stores a model at the modelFolder or in tmp folder
cpdef void storeSameModel(model) except *:
    cdef str fileName
    cdef str dirName
    
    dirName = conf.MODELS_DIR + "/"
    fileName = model.name
    
    outputFile = open(dirName + fileName, "wb")
    pickle.dump(model, outputFile, pickle.HIGHEST_PROTOCOL)
    outputFile.close()

# Clears the ./tmp/ folder
cpdef void clearTmpFolder() except *:
    for file in os.listdir(conf.TMP_DIR):
        os.remove(conf.TMP_DIR + "/" + file)

# Preprocessing with Spectrum
cpdef np.ndarray[CUINT64_t] processSpectrum(np.ndarray[CINT16_t] frame):
    cdef np.ndarray tmp = fft(frame)
    tmp = np.absolute(np.split(tmp, 2)[0])
    tmp = np.float64(tmp)
    return extractFeatures(tmp)

# Preprocessing with Cepstrum
cpdef np.ndarray[CUINT64_t] process(np.ndarray[CINT16_t] frame):
    cdef Py_ssize_t i
    cdef np.ndarray[CUINT64_t] result = np.zeros(32, dtype=np.float64)
    cdef np.ndarray tmp = fft(frame)
    tmp = np.absolute(tmp)
    tmp[0] *= tmp[0]
    for i in range(1, 513):
        tmp[i] *= tmp[i]
        tmp[1024 - i] = tmp[i]
    tmp = np.log(tmp)
    tmp = ifft(tmp)
    tmp = np.absolute(np.split(tmp, 2)[0])
    for i in range(512):
        result[i/16] += math.pow(tmp[i], 2)
    return result


cdef np.ndarray[CUINT64_t] extractFeatures(np.ndarray[CUINT64_t] in_array):
    cdef Py_ssize_t i
    cdef np.ndarray[CUINT64_t] result = np.empty(64, dtype=np.float64)
    cdef Py_ssize_t k = 0
    #print("length in_array: " + str(len(in_array)) + " | length result: " + str(len(result)))
    for i in range(len(in_array)):
        result[k%16] += in_array[i]
        k += 1
    #print("Type of: " + str(type(result[0])))
    return result