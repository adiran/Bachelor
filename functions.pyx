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
            tmp1 = <CUINT64_t>abs(recordData[i])# * in_array1[i])
            #print("Test1 " + str(in_array1[i]) + " | " + str(in_array2[i]))
            tmp2 = <CUINT64_t>abs(modelData[i])# * in_array2[i])
            #print("Test2 " + str(result) + " | " + str(abs(tmp1 - tmp2)) + " | " + str(result + abs(tmp1 - tmp2)))
            if tmp1 > tmp2:
                #print("Test2 " + str(result) + " | " + str(abs(tmp1 - tmp2)) + " | " + str(result + abs(tmp1 - tmp2)))
                result += tmp1 - tmp2
            else:
                #print("Test2 " + str(result) + " | " + str(abs(tmp2 - tmp1)) + " | " + str(result + abs(tmp2 - tmp1)))
                result += tmp2 - tmp1
            #print("Test3")
    else:
        print("Compare failed because of different shapes")
        result = 18446744073709551615
    return result


cpdef tuple minimalizeAndCalcTolerance(list in_array, int optimalFrames, int iteration):
    in_array = copy.deepcopy(in_array)
    cdef list frameMember = []
    cdef list diffrences = []
    cdef CUINT64_t minDiffrence = 0.
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
    for i in range(length):
        frameMember.append(i)
        if i < length -1:
            diffrences.append(compare(in_array[i], in_array[i+1]))
    while length > optimalFrames:
        minDiffrence = np.finfo(np.float64).max - np.finfo(np.float64).eps
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
            if diffrences[i] < minDiffrence:
                #print("diffrences[" + str(i) + "]: " + str(diffrences[i]) + " | minDiffrence: " + str(minDiffrence))
                pos = i
                minDiffrence = diffrences[i]
        if minDiffrence > maxDiffrence:
            maxDiffrence = minDiffrence
        #print("Set frameMember[" + str(pos+1) +"] from " + str(frameMember[pos+1]) + " to " + str(frameMember[pos]) + ".")
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
    #for i in range(optimalFrames - 1):
        #print("Diffrences:\t" + str(compare(result[i], result[i+1])))
    print("maxDiffrences:\t" + str(maxDiffrence))
    return (result, maxDiffrence)

cpdef tuple minimalizeAndCalcTolerance1(list in_array, int optimalFrames, int iteration):
    in_array = copy.deepcopy(in_array)
    cdef list counter = [1 for x in range(len(in_array))]
    cdef list tmpCounter = []
    cdef list diffrences = []
    cdef list result = []
    cdef np.ndarray[CUINT64_t] tmp
    cdef CUINT64_t minDiffrence
    cdef CUINT64_t maxDiffrence = 0
    cdef int pos
    cdef Py_ssize_t i
    while len(in_array) > optimalFrames:
        for i in range(len(in_array) - 1):
            diffrences.append(compare(in_array[i]/counter[i], in_array[i+1]/counter[i+1]))
        minDiffrence = diffrences[0]
        pos = 0
        for i in range(len(diffrences)):
            if diffrences[i] < minDiffrence:
                pos = i
                minDiffrence = diffrences[i]
        i = 0
        while i < len(in_array):
            if i == pos:
                if minDiffrence > maxDiffrence:
                    maxDiffrence = minDiffrence
                tmp = in_array[i]
                tmpCounter.append(counter[i] + counter[i+1])
                for j in range(tmp.size):
                    tmp[j] = (tmp[j] + in_array[i+1][j])
                result.append(copy.deepcopy(tmp))
                i += 2
            else:
                result.append(copy.deepcopy(in_array[i]))
                tmpCounter.append(counter[i])
                i += 1
        in_array = copy.deepcopy(result)
        counter = copy.deepcopy(tmpCounter)
        diffrences = []
        tmpCounter = []
        result = []
    for i in range(len(in_array)):
        result.append(copy.deepcopy(in_array[i]/counter[i]))
        if i < len(in_array)-1:
            diffrences.append(compare(in_array[i], in_array[i+1]))
    minDiffrence = diffrences[0]
    for i in range(len(diffrences)):
        if diffrences[i] < minDiffrence:
            minDiffrence = diffrences[i]
        #if iteration < 10:
            #print(str(iteration) + " | Diffrences:\t\t" + str(diffrences[i]))
        #else: 
            #print(str(iteration) + " | Diffrences:\t" + str(diffrences[i]))
    print(str(iteration) + " | maxDiffrence:\t" + str(maxDiffrence))
    return (result, maxDiffrence)


# minimalizes the features by comparing. Builds the average over all
# similar frames
cdef list minimalizeFeatures(list in_array, CUINT64_t tolerance, int optimalFrames, int iteration):
    in_array = copy.deepcopy(in_array)
    #print(str(iteration) + "mF | Length in_array: " + str(len(in_array)))
    cdef list counter = [1 for x in range(len(in_array))]
    cdef list tmpCounter = []
    cdef list diffrences = []
    cdef list result = []
    cdef CUINT64_t minDiffrence
    cdef int pos
    cdef bint recalculate = True
    cdef Py_ssize_t i
    while recalculate:
        #print("Length in_array: " + str(len(in_array)) + " | optimalFrames: " + str(optimalFrames))
        diffrences = []
        #print(str(iteration) + "mF 1")
        for i in range(len(in_array) - 1):
            diffrences.append(compare(in_array[i]/counter[i], in_array[i+1]/counter[i+1]))
        #print(str(iteration) + "mF " + str(len(diffrences)))
        minDiffrence = diffrences[0]
        pos = 0
        
        #print(str(iteration) + "mF 2")
        for i in range(len(diffrences)):
            if diffrences[i] < minDiffrence:
                pos = i
                minDiffrence = diffrences[i]
        #print(str(iteration) + "mF 3")

        if minDiffrence > tolerance:
            recalculate = False
            break
        #print(str(iteration) + "mF 4")
        i = 0
        while i < len(in_array):
            if i == pos:
                tmp = in_array[i]
                tmpCounter.append(counter[i] + counter[i + 1])
                for j in range(tmp.size):
                    tmp[j] = (tmp[j] + in_array[i+1][j])
                result.append(copy.deepcopy(tmp))
                i += 2
            else:
                result.append(in_array[i])
                tmpCounter.append(counter[i])
                i += 1
        in_array = copy.deepcopy(result)
        #print("1Length in_array: " + str(len(in_array)) + " | optimalFrames: " + str(optimalFrames))
        counter = copy.deepcopy(tmpCounter)
        tmpCounter = []
        result = []
        if len(in_array) <= optimalFrames:
            recalculate = False
    #print("Length : " + str(len(result)))
    for i in range(len(in_array)):
        result.append(copy.deepcopy(in_array[i]/counter[i]))
    return (result)

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

# merges two models
cpdef void modelMergeNew(tuple data) except *:
    cdef list in_array = data[0]
    cdef list minimalizedRecords = []
    cdef int optimalFrames = data[1]
    cdef str name = data[2]
    cdef int iteration = data[3]
    cdef list result
    cdef CUINT64_t tolerance
    cdef Py_ssize_t i
    cdef Py_ssize_t pos
    cdef Py_ssize_t j
    #print(str(iteration) + " | process model that starts with " + str(in_array[0][0][0]))
    in_array[0], tolerance = minimalizeAndCalcTolerance(in_array[0], optimalFrames, iteration)
    #print(str(iteration) + " | Tolerance calculated: " + str(tolerance))
    if tolerance != 0:
        length = len(in_array)
        #print(str(iteration) + " | length: " + str(length))
        for i in range(1, length):
            tmp = minimalizeFeatures(in_array[i], tolerance, optimalFrames, i)
            #print(str(iteration) + " | i: " + str(i))
            if len(tmp) > 0:
                minimalizedRecords.append(tmp)
        length = len(minimalizedRecords)
        result = copy.deepcopy(in_array[0])
        # counts on which position we are in a specific array
        counter = np.zeros(length, dtype=int)
        #print("Length result: " + str(len(result)))
        # counts how many arrays influence the result frame
        counterResult = np.ones(len(result), dtype=int)
        # for every frame in result try if we find mergable frames
        for pos in range(len(result)):
            # we try all recordings
            for i in range(length):
                # we only try frames after the last merged frame from this
                # recording
                for j in range(counter[i], len(minimalizedRecords[i])):
                    # if the average over all merged frames compared to the current
                    # frame of the current model is under tolerance we merge the
                    # current frame to the others
                    if compare(np.asarray(in_array[0][pos], dtype=np.float64), minimalizedRecords[i][j]) < tolerance:
                        result[pos] = np.add(result[pos], in_array[i + 1][j])
                        counterResult[pos] += 1
                        counter[i] = j
        #print(str(iteration) + " | Sum calculated")
        # calculate the average from the stored sum
        for pos in range(len(result)):
            result[pos] = np.asarray([(x / counterResult[pos]) for x in result[pos]], dtype=np.float64)
        #print(str(iteration) + " | Average calculated")
        tolerance /= conf.TOLERANCE_MULTIPLIER
        print("Merged iteration " + str(iteration) + ", with " + str(length) + " records | Tolerance: " + str(tolerance))
        storeModel(model.Model(result, tolerance, name, length), True, iteration)


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
    cdef np.ndarray[CUINT64_t] result = np.zeros(64, dtype=np.float64)
    cdef Py_ssize_t k = 0
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
        result[k%16] += math.pow(tmp[i], 2)
        k += 1
    result = np.float64(result)
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