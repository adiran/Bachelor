import wave
import numpy as np
import functions as f
from scipy.fftpack import fft, ifft
import config as conf
import pyaudio
import struct
import copy
import scipy.io.wavfile as scwave

p = pyaudio.PyAudio()
RATE = int(p.get_device_info_by_index(conf.DEVICE_INDEX)['defaultSampleRate'])

backgroundNoise = np.zeros(1024, dtype=np.complex128)
wf = wave.open("records/microwave/backgroundNoise.wav")
sampwidth = wf.getsampwidth()
loops = int(wf.getnframes() / 1024)
for i in range(loops):
    framesAsString = wf.readframes(1024)
    frame = np.fromstring(framesAsString, np.int16)
    frame = fft(frame)
    for j in range(1024):
        backgroundNoise[j] += frame[j] / loops


wf = wave.open("records/microwave/1.wav")
loops = int(wf.getnframes() / 1024)
noiseFree = []
for i in range(loops):
    #print(i)
    framesAsString = wf.readframes(1024)
    frame = np.fromstring(framesAsString, np.int16)
    print(str(frame))
    frame = fft(frame)
    for j in range(1024):
        frame[j] -= backgroundNoise[j]
    noiseFree.append(frame)
print("--------------------------------------------------------------------------------------------")
result = np.empty(loops*1024, dtype=np.int16)
for i in range(loops):
    noiseFree[i] = np.int16(ifft(noiseFree[i]).real)
    print(str(noiseFree[i]))
    for j in range(1024):
        result[i*1024 + j] = noiseFree[i][j]
scwave.write("tmp/tmp.wav", RATE, result)
