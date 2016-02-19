"""Audio Trainer v1.0"""
import pyaudio
import wave
import time
import sys
import struct
import math
import audioop
import config as conf
import numpy as np
import functions as f
from scipy.fftpack import fft

import listenC

def callback(in_data, frame_count, time_info, status):
    #if status > 0:
        #print("Status: " + str(status))
    listenC.listen(in_data)
    return (None, pyaudio.paContinue)

def main():

    # instantiate PyAudio
    p = pyaudio.PyAudio()
    
    # get default sample rate
    RATE = int(p.get_device_info_by_index(conf.DEVICE_INDEX)['defaultSampleRate'])

    # setup the listenC
    listenC.setup()

    raw_input("Press Enter key to start listening")

    # wait to prevent to capture the hitting of enter key
    time.sleep(.08)

    # TODO just for testing, set up time
    listenC.beginning()

    # open audio stream with callback function
    stream = p.open(format=conf.FORMAT,
                    channels=conf.CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=conf.CHUNK,
                    stream_callback=callback,
                    input_device_index=conf.DEVICE_INDEX)

    print("* listening")

    # start audio stream
    stream.start_stream()

    # wait for stream to finish
    while stream.is_active():
        raw_input("Press Enter key to stop listening")
        stream.stop_stream()
    
    # close stream
    stream.close()
    p.terminate()
