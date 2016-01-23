import pyaudio
import wave
import sys
import config as conf


def callback(in_data, frame_count, time_info, status):
    data = None
    return (data, pyaudio.paContinue)

p = pyaudio.PyAudio()
RATE = int(p.get_device_info_by_index(conf.DEVICE_INDEX)['defaultSampleRate'])

CHUNK = 1024




wf = wave.open("records/asd/1.wav")

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=conf.FORMAT,
                channels=conf.CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=conf.CHUNK,
                stream_callback=callback,
                input_device_index=conf.DEVICE_INDEX)

stream.start_stream()

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()