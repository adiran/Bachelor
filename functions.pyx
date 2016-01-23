import numpy as np
cimport numpy as np
from scipy.fftpack import rfft
import copy
import os
import pickle
import time
import random
from features import mfcc
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
CUINT32 = np.uint32
ctypedef np.uint32_t CUINT32_t

cpdef np.ndarray[CUINT32_t] extractFeatures(np.ndarray[CUINT32_t] in_array):
    cdef Py_ssize_t i
    cdef np.ndarray[CUINT32_t] result = np.zeros(64, dtype=np.uint32)
    cdef CUINT32_t j = 0
    cdef CUINT32_t k = 0
    cdef CUINT32_t h = 0
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


cpdef CUINT64_t compare(np.ndarray[CUINT32_t] in_array1, np.ndarray[CUINT32_t] in_array2):
    cdef CUINT64_t result = 0
    cdef Py_ssize_t i
    cdef CUINT64_t tmp1
    cdef CUINT64_t tmp2
    if len(in_array1) == len(in_array2):
        for i in range(len(in_array1)):
            tmp1 = <CUINT64_t>abs(in_array1[i] * in_array1[i])
            #print("Test1 " + str(in_array1[i]) + " | " + str(in_array2[i]))
            tmp2 = <CUINT64_t>abs(in_array2[i] * in_array2[i])
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


# minimalizes the features by comparing. Builds the average over all
# similar frames


cdef list minimalizeFeatures(list in_array, CUINT64_t tolerance, int iteration):
    in_array = copy.deepcopy(in_array)
    #print(str(iteration) + "mF | Length in_array: " + str(len(in_array)))
    cdef list result = [in_array[0]]
    cdef int count = 1
    cdef np.ndarray[CUINT32_t] currentFeatureVector 
    cdef Py_ssize_t i
    tolerance = tolerance * conf.TOLERANCE_MULTIPLIER
    for i in range(len(in_array) - 1):
        currentFeatureVector = result.pop()
        #print("Compare(): " + str(compare(np.asarray([(x / count) for x in currentFeatureVector], dtype = np.uint64), in_array[i + 1]) / 10000000000) + " | tolerance: " + str(tolerance / 10000000000))
        if compare(np.asarray([(x / count) for x in currentFeatureVector], dtype = np.uint32),
                   in_array[i + 1]) < tolerance:
            currentFeatureVector = np.add(
                currentFeatureVector, in_array[i + 1])
            result.append(currentFeatureVector)
            count += 1
        else:
            currentFeatureVector = np.asarray([x / count for x in currentFeatureVector], dtype = np.uint32)
            result.append(currentFeatureVector)
            result.append(in_array[i + 1])
            count = 1
    currentFeatureVector = result.pop()
    currentFeatureVector = np.asarray([x / count for x in currentFeatureVector], dtype = np.uint32)
    result.append(currentFeatureVector)
    return result

# loads a previously stored model
cpdef np.ndarray loadModels(bint tmp = False):
    cdef list result =  []
    cdef str directory
    if tmp:
        directory = conf.TMP_DIR
    else:
        directory = conf.MODELS_DIR
    for file in os.listdir(directory):
        openFile = open((directory + "/" + file), "rb")
        result.append(pickle.load(openFile))
        openFile.close()
    return np.asarray(result)

# merges two models
cpdef void modelMergeNew(tuple data):
    cdef list in_array = data[0]
    cdef int minFrames = data[1]
    cdef int maxFrames = data[2]
    cdef str name = data[3]
    cdef int iteration = data[4]
    cdef list result
    cdef Py_ssize_t i
    cdef Py_ssize_t pos
    cdef Py_ssize_t j

    #print(str(iteration) + " | process model that starts with " + str(in_array[0][0][0]))
    tolerance = calculateTolerance(in_array[0], minFrames, maxFrames, iteration)
    print(str(iteration) + " | Tolerance calculated: " + str(tolerance))
    # if the tolerance is below 1
    if tolerance != 0:
        length = len(in_array)
        for i in range(length):
            in_array[i] = minimalizeFeatures(in_array[i], tolerance, iteration)
        result = in_array[0]
        # counts on which position we are in a specific array
        counter = np.zeros(length, dtype=int)
    #    print("Length result: " + str(len(result)))
        # counts how many arrays influence the result frame
        counterResult = np.ones(len(result), dtype=int)
        # for every frame in result try if we find mergable frames
        for pos in range(len(result)):
            # we try all recordings
            for i in range(length - 1):
                # we only try frames after the last merged frame from this
                # recording
                for j in range(counter[i + 1], len(in_array[i + 1])):
                    # if the average over all merged frames compared to the current
                    # frame of the current model is under tolerance we merge the
                    # current frame to the others
                    if compare(np.asarray([(x / counterResult[pos]) for x in result[pos]], dtype=np.uint32), in_array[i + 1][j]) < tolerance:
                        result[pos] = np.add(result[pos], in_array[i + 1][j])
                        counterResult[pos] += 1
                        counter[i + 1] = j
        #print(str(iteration) + " | Sum calculated")
        # calculate the average from the stored sum
        for pos in range(len(result)):
            result[pos] = np.asarray([(x / counterResult[pos]) for x in result[pos]], dtype=np.uint32)
        #print(str(iteration) + " | Average calculated")
        tolerance /= conf.TOLERANCE_MULTIPLIER
        print("Merged iteration " + str(iteration) + ", length: " + str(len(result)) + " | Tolerance: " + str(tolerance))
        storeModel(model.Model(result, tolerance, name), True, iteration)

# stores a model at the modelFolder or in tmp folder
cpdef void storeModel(model, bint tmp=False, int iteration=0):
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

    outputFile = open(dirName + fileName, "wb")
    pickle.dump(model, outputFile, pickle.HIGHEST_PROTOCOL)
    outputFile.close()

# Clears the ./tmp/ folder
cpdef void clearTmpFolder():
    for file in os.listdir(conf.TMP_DIR):
        os.remove(conf.TMP_DIR + "/" + file)


# Calculates the tolerance for a model by the given first record
cdef CUINT64_t calculateTolerance(list record, int minFrames, int maxFrames, int iteration):
    cdef CUINT64_t tolerance = conf.TOLERANCE
    # print(str(iteration) + " | Length record " + str(len(record)) + " | beginns with: " + str(record[0]))
    cdef list first = minimalizeFeatures(record, tolerance, iteration)
    # print(str(iteration) + " | Length first" + str(len(first)))
    cdef CUINT64_t change = 1000000000000
    cdef bint recalculate = True
    cdef bint switch
    if len(first) < minFrames:
        switch = True
    elif len(first) > maxFrames:
        switch = False
    else:
        recalculate = False
    # calculate the tolerance for the given min and max frames
    while recalculate:
        if tolerance > 0:
            # print(str(iteration) + " | Length: " + str(len(first)) + " | tolerance: " + str(tolerance))
            if len(first) < maxFrames:
                if len(first) < minFrames:
                    if switch == False:
                        switch = True
                        if change > 1:
                            change /= 10
                        else:
                            print("Error, its not possible to get the length of this model fit in the borders.")
                            #debugCalculateTolerance(record, tolerance, iteration)
                            return 0
                    while change > tolerance:
                        change /= 10
                    tolerance -= change
                    first = minimalizeFeatures(record, tolerance, iteration)
                else:
                    recalculate = False
            else:
                if switch:
                    switch = False
                    if change > 1:
                        change /= 10
                    else:
                        print("Error, its not possible to get the length of this model fit in the borders.")
                        #debugCalculateTolerance(record, tolerance, iteration)
                        return 0
                tolerance += change
                first = minimalizeFeatures(record, tolerance, iteration)
        else:
            print("Error, tolerance < 0")
            return 0
    return tolerance


cpdef np.ndarray[CUINT32_t] process(np.ndarray[CINT16_t] frame):
    
    # TODO baue das generisch
    #cdef int rate = 44100
    cdef np.ndarray tmp = rfft(frame)
    #tmp = np.abs(tmp)
    # needed for rfft
    tmp = np.uint32(tmp)
    return tmp
    #np.split(np.uint64(np.abs(fft(frame))), 2)[0]

cdef void debugCalculateTolerance(list record, CUINT64_t tolerance, int iteration):
    file = open("debug/calculateTolerance" + str(iteration), "w")
    cdef list first = minimalizeFeatures(record, tolerance, iteration)
    file.write("Length record: " + str(len(record)) + " | Length minimalized: " + str(len(first)))
    file.write("\n")
    for i in range(len(record) - 1):
        debugCompare(record[i], record[i+1], file, tolerance, i)
    file.close()

cpdef void debugCompare(np.ndarray[CUINT32_t] in_array1, np.ndarray[CUINT32_t] in_array2, file, CUINT64_t tolerance, frameNumber):
    cdef CUINT64_t result = 18446744073709551615
    cdef Py_ssize_t i
    cdef CUINT64_t tmp1 = 0
    cdef CUINT64_t tmp2 = 0
    if len(in_array1) == len(in_array2):
        result = 0
        for i in range(len(in_array1)):
            tmp1 += <CUINT64_t>abs(in_array1[i] * in_array1[i])
            tmp2 += <CUINT64_t>abs(in_array2[i] * in_array2[i])
            result = tmp1 + tmp2
        result = tmp1 + tmp2
        file.write(str(frameNumber) + ": " + str(tmp1) + " + " + str(tmp2) + " = " + str(result) + " | " + str(result) + " - " + str(tolerance) + " = " + str(result - tolerance))
        file.write("\n")
        file.flush()
    else:
        print("Compare failed because of different shapes")