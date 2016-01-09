"""Audio Trainer v1.0"""
import pyaudio
import wave
import time
import sys
import struct
import math
import audioop
import os.path
import config as conf
import numpy as np
import functions as f

# instantiate PyAudio
p = pyaudio.PyAudio()

# get default sample rate
RATE = int(p.get_device_info_by_index(conf.DEVICE_INDEX)['defaultSampleRate'])

# preprocessed audio data
number = []

# wave data for output and testing
waveFrames = []

# we store 2 frames of collected audio data as 1 frame in sample so here
# is a switch
switch = True

# used for storing half of a frame because of the switch
frame = []

# Just used for testing
testframes = []


# used to identify if and how many frames should be captured althoug the
# audio level is under threshold
framesAfterSound = 0
framesSwitch = True

# counter for waveoutput in multiple files
wavenumber = conf.WAVENUMBER

# define callback for PyAudio


def callback(in_data, frame_count, time_info, status):
    global framesAfterSound
    global waveFrames
#    print("in_data:")
#    print(in_data)
#    print("frame_count")
#    print(frame_count)
#    print("time_info")
#    print(time_info)
#    print("status")
#    print(status)
    # calculate the audio level for sounddetection
#    print("audioLevel = math.sqrt(abs(audioop.avg(in_data, 4)))")
    audioLevel = math.sqrt(abs(audioop.avg(in_data, 4)))
    waveFrames.append(in_data)

    # TODO Just for testing
    testframes.append(np.fromstring(in_data, np.int16))

    framesSwitch = True
#    print("mich solltest du nicht mehr sehen")

    # if audio level is under conf.THRESHOLD but we had sound in the frames before we capture a few frames more to prevent single missing frames
    # first we check the audio level because framesAfterSound > 0 occures way
    # more than aduioLevel <= conf.THRESHOLD
    if audioLevel <= conf.THRESHOLD:
        #        print("1")
        if framesAfterSound > 0:
            #            print("2")
            audioLevel = conf.THRESHOLD + 1
#            print("3")
            framesAfterSound -= 1
#            print("4")
            framesSwitch = False

#    print("5")
    # the whole capturing and preprocessing
    if audioLevel > conf.THRESHOLD:
        global switch
        global frame
#        print("6")

        # if framesAfterSound has been decrement but audio level rised over conf.THRESHOLD again we reset framesAfterSound
        # first we check if we decreased framesAfterSound because framesSwitch
        # is True by default
        if framesAfterSound < conf.FRAMES_AFTER_SOUND:
            if framesSwitch:
                #                print("7")
                framesAfterSound = conf.FRAMES_AFTER_SOUND

        # capturing the first half of the frame
        if switch:
            #            print("frame = np.fromstring(in_data, np.int16)")
            frame = np.fromstring(in_data, np.int16)
#            print("waveFrames.append(in_data)")
            switch = False

        # capturing the second half and preprocessing
        else:
            global number
#            print("frame = np.append(frame, np.fromstring(in_data, np.int16))")
            frame = np.append(frame, np.fromstring(in_data, np.int16))
#            print("number.append(f.extractFeatures(data[0]))")
            # TODO find out which is the best solution for training a model
            # f.extractFeatures(data[0]))
            number.append(f.process(frame))
#            print("waveFrames.append(in_data)")
            switch = True

    data = None
#    print("return (data, pyaudio.paContinue)")
    return (data, pyaudio.paContinue)


def trainer(modelLocation):
    global number
    global waveFrames
    global testframes
    isUserInputNotANumber = True
    recordDuration = 0
    while isUserInputNotANumber:
        userInput = input("How many seconds should be captured? ")
        try:
            recordDuration = int(userInput)
            if recordDuration > 0:
                isUserInputNotANumber = False
            else:
                print("Duration should be greater 0")
        except ValueError:
            print("That was not a number")

    # open audio stream with callback function
    stream = p.open(format=conf.FORMAT,
                    channels=conf.CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=conf.CHUNK,
                    stream_callback=callback,
                    input_device_index=conf.DEVICE_INDEX)

    userInput = ''  # TODO t'
    while userInput == '':  # TODO t':
        print("Ready to record %d seconds" % (recordDuration))
        print(
            "Please note that there is a delay of 0,08 seconds before the recording start to prevent to capture you hitting the Enter key")
        input("Press Enter key to start recording")

        # wait to prevent to capture the hitting of enter key
        time.sleep(.08)
#        print("1")

        number = []
        waveFrames = []
        testframes = []

        # start audio stream
        stream.start_stream()

        print("* recording")

        # wait for stream to finish
        while stream.is_active():
            time.sleep(recordDuration)
            stream.stop_stream()

        print("* done recording")
        print()
        print("Länge number:" + str(len(number)))
        print()
        print("Länge waveFrames:" + str(len(waveFrames)))
        userInput = input(
            "Press Enter to add that record to your model or input anything for not adding it")
        if userInput == '':
            global wavenumber
            # TODO just for testing. remove before finish
            # write fft data to file
            file = open("array" + str(wavenumber), "w")
            file.write(str(len(testframes)))
            file.write("\n")
            for i in range(len(testframes)):
                file.write(", ".join(str(e) for e in testframes[i]))
                file.write("\n")
                file.flush()
            file.close()

            # TODO just for testing. remove before finish
            # write changes of frames to file
            file = open("changes", "w")
            file.write(str(len(number)))
            file.write("\n")
            count = 0
            for i in range(len(number) - 1):
                line = f.compare(number[i], number[i+1])
                if line < conf.TOLERANCE:
                    count += 1
                file.write(str(line))
                file.write("\n")
                file.flush()
            file.write(str(count))
            file.flush()
            file.close()

            # minimalize features
            number = f.minimalizeFeatures(number)

            print("Länge number:" + str(len(number)))
            print()

            # TODO just for testing. remove before finish
            # write minimalized features to file
            file = open("minimalizedFeatures", "w")
            file.write(str(len(number)))
            file.write("\n")
            for i in number:
                file.write(", ".join(str(e) for e in i))
                file.write("\n")
            file.close()

            # TODO just for testing. remove before finish
            # write audio data to wave file
            wf = wave.open("output" + str(wavenumber) + ".wav", "wb")
            wavenumber = wavenumber + 1
            wf.setnchannels(conf.CHANNELS)
            wf.setsampwidth(p.get_sample_size(conf.FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(waveFrames))
            wf.close

            # if this model is already trained we merge the old one and the new
            # data
            if os.path.isfile(modelLocation):
                numberOld = np.load(modelLocation)
                number = f.modelMerge(number, numberOld)

            np.save(modelLocation, number)

        userInput = input(
            "Please type t to (t)rain or anything else to quit. ")

    # close stream
    stream.close()
    p.terminate()

    print("Quit succefull")
    return("0")

# TODO just for testing. Remove before finish
# if this script is run directly
trainer("/home/pi/v1/model1.npy")
