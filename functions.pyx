import numpy as np
cimport numpy as np
from scipy.fftpack import fft
import copy
import os
import pickle
import time
import random
#from features import mfcc
import sys
import pdb

#own imports
import config as conf
import trainRecorded
import model
import interactions

# Cython stuff
CFLOAT = np.float64
ctypedef np.float64_t CFLOAT_t
CUINT64 = np.uint64
ctypedef np.uint64_t CUINT64_t
CINT16 = np.int16
ctypedef np.int16_t CINT16_t


cdef np.ndarray[CUINT64_t] extractFeatures(np.ndarray[CUINT64_t] in_array):
    cdef Py_ssize_t i
    cdef np.ndarray[CUINT64_t] result = np.zeros(64, dtype=np.uint64)
    cdef CUINT64_t j = 0
    cdef CUINT64_t k = 0
    cdef CUINT64_t h = 0
    #print("length in_array: " + str(len(in_array)) + " | length result: " + str(len(result)))
    for i in in_array:
        result[j] += i
        k += 1
        h += 1
        if(k == 16):
            j += 1
            k = 0
    #print("Type of: " + str(type(result[0])))
    return result

# summe der quadrate der distanzen zweier int arrays


cpdef CUINT64_t compare(np.ndarray[CUINT64_t] in_array1, np.ndarray[CUINT64_t] in_array2):
    cdef CUINT64_t result = 0
    cdef Py_ssize_t i
    cdef CUINT64_t tmp1
    cdef CUINT64_t tmp2
    if len(in_array1) == len(in_array2):
        for i in range(len(in_array1)):
            tmp1 = <CUINT64_t>abs(in_array1[i])# * in_array1[i])
            #print("Test1 " + str(in_array1[i]) + " | " + str(in_array2[i]))
            tmp2 = <CUINT64_t>abs(in_array2[i])# * in_array2[i])
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
    #print(str(iteration) + "mF | Length in_array: " + str(len(in_array)))
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
        #print(str(iteration) + "mF | Length in_array: " + str(len(in_array)) + " | optimalFrames: " + str(optimalFrames))
        for i in range(len(in_array) - 1):
            diffrences.append(compare(in_array[i]/counter[i], in_array[i+1]/counter[i+1]))
        minDiffrence = diffrences[0]
        pos = 0
        for i in range(len(diffrences)):
            #print(str(iteration) + "mF | maxDiffrence: " + str(diffrences[i]))
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
    #print(str(iteration) + "mF | maxDiffrence: ")
    for i in range(len(in_array)):
        result.append(copy.deepcopy(in_array[i]/counter[i]))
        if i < len(in_array)-1:
            diffrences.append(compare(in_array[i], in_array[i+1]))
        #print("Test1")
    minDiffrence = diffrences[0]
    for i in range(len(diffrences)):
        if diffrences[i] < minDiffrence:
            minDiffrence = diffrences[i]
        if iteration < 10:
            print(str(iteration) + " | Diffrences:\t\t" + str(diffrences[i]))
        else: 
            print(str(iteration) + " | Diffrences:\t" + str(diffrences[i]))
    print(str(iteration) + " | maxDiffrence:\t" + str(maxDiffrence))
    #if minDiffrence > maxDiffrence * 2:
    #    maxDiffrence = minDiffrence - (maxDiffrence * 2)
    #print(str(iteration) + " | maxDiffrence:\t" + str(maxDiffrence))
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
                    if compare(np.asarray(in_array[0][pos], dtype=np.uint64), minimalizedRecords[i][j]) < tolerance:
                        result[pos] = np.add(result[pos], in_array[i + 1][j])
                        counterResult[pos] += 1
                        counter[i] = j
        #print(str(iteration) + " | Sum calculated")
        # calculate the average from the stored sum
        for pos in range(len(result)):
            result[pos] = np.asarray([(x / counterResult[pos]) for x in result[pos]], dtype=np.uint64)
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


cpdef np.ndarray[CUINT64_t] process(np.ndarray[CINT16_t] frame):
    
    # TODO baue das generisch
    #cdef int rate = 44100
    cdef np.ndarray tmp = fft(frame)
    #print("tmp[0]: " + str(tmp[0]) + " | type of tmp[0]: " + str(type(tmp[0])))
    # needed for rfft
    # TODO ist das sinnvoll?
    tmp = np.absolute(np.split(tmp, 2)[0])
    tmp = np.uint64(tmp)
    #print("tmp[0]: " + str(tmp[0]) + " | type of tmp[0]: " + str(type(tmp[0])))
    tmp = extractFeatures(tmp)
    #for i in range(tmp.size):
        #tmp[i] = tmp[i]*tmp[i]
    return tmp
    #np.split(np.uint64(np.abs(fft(frame))), 2)[0]