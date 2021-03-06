"""Audio Trainer v1.0"""
# import of own scripts
import os
import pyaudio
import time

# import of own scripts
import config as conf

# instantiate PyAudio
p = pyaudio.PyAudio()

# get default sample rate
RATE = int(p.get_device_info_by_index(conf.DEVICE_INDEX)['defaultSampleRate'])

# get all the parameters for training
def getTrainParameters():
    records = []
    isUserInputNotANumber = True
    selectedOption = 0
    while isUserInputNotANumber:
        i = 0
        for file in os.listdir(conf.RECORDS_DIR):
            if os.path.isdir(conf.RECORDS_DIR + "/" + file):
                records.append(file)
                i += 1
                print("\t" + str(i) + ": " + file)
        userInput = raw_input("Which option do you want to choose? ")
        try:
            selectedOption = int(userInput)
            if selectedOption > 0:
                if selectedOption <= i:
                    isUserInputNotANumber = False
                else:
                    print("There are only " + str(i) + " options.")
            else:
                print("There are nether negative options nor an option 0.")
        except ValueError:
            print("That was not a number")

    recordName = records[selectedOption - 1]
    recordNumber = 1
    fileName = conf.RECORDS_DIR + "/" + recordName
    while os.path.isfile(fileName + "/" + str(recordNumber) + ".wav"):
        recordNumber += 1
    recordNumber -= 1

    print("What name should your model have?")
    modelName = recordName
    userInput = raw_input(
        "Input name of model or nothing for the same name as the records (" +
        str(recordName) +
        ")?")
    if userInput != "":
        modelName = userInput

    print()
    print(
        "How many frames should this model at optimum have?")
    isUserInputNotANumber = True
    while isUserInputNotANumber:
        userInput = raw_input(
            "Input number of frames: ")
        if userInput != "":
            try:
                optimalFrames = int(userInput)
                if optimalFrames > 1:
                    isUserInputNotANumber = False
                else:
                    print("You should have more than 1 Frames.")
            except ValueError:
                print("That was not a number.")

    print()
    print("Where is the python script that should be executed after recognizing this model?.")
    isUserInputWrong = True
    scriptpath = None
    while isUserInputWrong:
        scriptpath = raw_input(
            "Path to the script file: ")
        if scriptpath != "":
            if os.path.isfile(str(scriptpath)):
                isUserInputWrong = False
            else:
                print("It seems that there is no file at " + str(scriptpath) + ".")
        else:
            print("Please type in a path.")

    return(fileName, modelName, optimalFrames, scriptpath)


def getModelNumber(maxModelNumber):
    print("Which model do you want to store and use?")
    isUserInputNotANumber = True
    while isUserInputNotANumber:
        userInput = raw_input("Model number: ")
        try:
            modelNumber = int(userInput)
            if modelNumber > maxModelNumber:
                print("There are only " + str(maxModelNumber) + " models so your input shouldn't be greater than this.")
            elif modelNumber > 0:
                isUserInputNotANumber = False
            else:
                print("A negativ number or 0 doesn't make sense here so please type a number greater than 0")
        except ValueError:
            print("That was not a number.")
    return modelNumber

def getDifferentModelFileName(fileName):
    print("The chosen filename (" + fileName + ") for this model has already been used.")
    tmpFileName = fileName
    while tmpFileName == fileName:
        tmpFileName = raw_input("Please type another filename: ")
        if tmpFileName == fileName:
            print("This is the same filename as the chosen filename. Please type another.")
    return tmpFileName

def getRecordParameters():
    print("Please select a sound to capture more records or create a new sound.")
    records = []
    isNewSound = False

    isUserInputNotANumber = True
    selectedOption = 0
    while isUserInputNotANumber:
        i = 1
        for file in os.listdir(conf.RECORDS_DIR):
            if os.path.isdir(conf.RECORDS_DIR + "/" + file):
                records.append(file)
                print("\t" + str(i) + ": " + file)
                i += 1
        print("\t" + str(i) + ": Create a new sound")
        userInput = raw_input("Which option do you want to choose? ")
        try:
            selectedOption = int(userInput)
            if selectedOption > 0:
                if selectedOption <= i:
                    isUserInputNotANumber = False
                else:
                    print("There are only " + str(i) + " options.")
            else:
                print("There are nether negative options nor an option 0.")
        except ValueError:
            print("That was not a number")

    if selectedOption < i:
        recordName = records[selectedOption - 1]
        recordNumber = 1
        while os.path.isfile(conf.RECORDS_DIR + "/" + recordName + "/" + str(recordNumber) + ".wav"):
            recordNumber += 1
    else:
        isNewSound = True
        isUserInputWrong = True
        while isUserInputWrong:
            recordName = raw_input("Which name should your sound have? ")
            isUserInputWrong = False
            for file in os.listdir(conf.RECORDS_DIR):
                if file == recordName:
                    isUserInputWrong = True
                    print("There is already a sound with that name")
        
        recordNumber = 1

    isUserInputNotANumber = True
    recordDuration = 0
    while isUserInputNotANumber:
        userInput = raw_input("How many seconds should be captured? ")
        try:
            recordDuration = int(userInput)
            if recordDuration > 0:
                isUserInputNotANumber = False
            else:
                print("Duration should be greater 0")
        except ValueError:
            print("That was not a number")

    if isNewSound:
        print("You created a new sound named " + recordName + " with a duration of " + str(recordDuration) + " seconds.")
    else:
        print("You selected " + recordName + " with a duration of " + str(recordDuration) + " seconds.")
    userInput = raw_input("Press enter to proceed or input anything to change your selection.")
    if userInput != "":
        return getRecordParameters()
    recordName = conf.RECORDS_DIR + "/" + recordName
    if isNewSound:
        os.mkdir(recordName)
    return (recordName, recordNumber, recordDuration)

# define callback (2)
def callback(in_data, frame_count, time_info, status):
    global wf
    global i

    if (i+1) < len(wf):
        data = wf[i] + wf[i+1]
    elif i < len(wf):
        data = wf[i]
    else:
        data = ""
    i += 2
    return (data, pyaudio.paContinue)

#returns an empty string if the audio data should be stored. 
def wantToStoreRecord(frames):
    global wf
    global i

    i = 0
    wf = frames
    stream = p.open(format=conf.FORMAT,
                    channels=conf.CHANNELS,
                    rate=RATE,
                    output=True,
                    stream_callback=callback)
    stream.start_stream()
    while stream.is_active():
        time.sleep(0.1)
    stream.stop_stream()
    stream.close()
    return raw_input("Press Enter to save that record or input anything for not saving it.")