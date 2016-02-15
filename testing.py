import pyaudio

p = pyaudio.PyAudio()

for i in range(10):
    print(p.get_device_info_by_index(i)['maxInputChannels'])