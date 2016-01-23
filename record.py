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
import interactions

# instantiate PyAudio
p = pyaudio.PyAudio()

# get default sample rate
RATE = int(p.get_device_info_by_index(conf.DEVICE_INDEX)['defaultSampleRate'])

# wave data for output and testing
waveFrames = []

# define callback for PyAudio


def callback(in_data, frame_count, time_info, status):
    global waveFrames
    waveFrames.append(in_data)
    data = None
    return (data, pyaudio.paContinue)


def main():
    global waveFrames
    global recordNumber
    dirName, recordNumber, recordDuration = interactions.getRecordParameters()

    userInput = ''  # TODO t'
    while userInput == '':  # TODO t':
        print("Ready to record %d seconds" % (recordDuration))
        print(
            "Please note that there is a delay of 0,08 seconds before the recording start to prevent to capture you hitting the Enter key")

        # open audio stream with callback function
        stream = p.open(format=conf.FORMAT,
                    channels=conf.CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=conf.CHUNK,
                    stream_callback=callback,
                    input_device_index=conf.DEVICE_INDEX)

        raw_input("Press Enter key to start " + str(recordNumber) + ". recording")

        # wait to prevent to capture the hitting of enter key
        time.sleep(.08)
#        print("1")

        waveFrames = []

        # start audio stream
        stream.start_stream()

        print("* recording")

        # wait for stream to finish
        while stream.is_active():
            time.sleep(recordDuration)
            stream.stop_stream()

        print("* done recording")
        print()
        
        # close stream
        stream.close()
        

        wf = wave.open("tmp/tmp.wav", "wb")
        recordNumber = recordNumber + 1
        wf.setnchannels(conf.CHANNELS)
        wf.setsampwidth(p.get_sample_size(conf.FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(waveFrames))
        wf.close

        userInput = interactions.wantToStoreRecord(waveFrames)
        f.clearTmpFolder()
        if userInput == '':
            # write audio data to wave file
            wf = wave.open(
                str(dirName) + "/" + str(recordNumber) + ".wav", "wb")
            recordNumber = recordNumber + 1
            wf.setnchannels(conf.CHANNELS)
            wf.setsampwidth(p.get_sample_size(conf.FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(waveFrames))
            wf.close

        userInput = raw_input(
            "Press Enter to record another record or input anything to quit.")

    
    p.terminate()

    print("Quit succefull")
    return("0")

