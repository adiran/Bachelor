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

import listenC as listen

def callback(in_data, frame_count, time_info, status):
    listen.listen(in_data)
    return (None, pyaudio.paContinue)

def main():

    # instantiate PyAudio
    p = pyaudio.PyAudio()
    
    # get default sample rate
    RATE = int(p.get_device_info_by_index(conf.DEVICE_INDEX)['defaultSampleRate'])

    # define callback for PyAudio

    input("Press Enter key to start listening")

    # wait to prevent to capture the hitting of enter key
    time.sleep(.08)

    # setup the listenC
    listen.setup()

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
        input("Press Enter key to stop listening")
        stream.stop_stream()
    
    # close stream
    stream.close()
    p.terminate()
