"""Audio Trainer v1.0"""
# import of own scripts
import record
import trainRecorded as train
import listen
import manageModels
import functions as f

print("Welcome to soundevent recognition on RasPi")
print("Setting up everything...")

f.clearTmpFolder()

userDoesntWantToQuit = True
while userDoesntWantToQuit:
    print("Available functions:")
    print("\t1\trecord sounds for an existing or a new model")
    print("\t2\ttrain a model")
    print("\t3\tlisten an recognize sounds")
    print("\t4\tmanage models")
    print("\t5\tquit")
    userInput = raw_input("What do you want to do?")
    try:
        selectedOption = int(userInput)
        if selectedOption > 0:
            if selectedOption == 1:
                record.main()
            elif selectedOption == 2:
                train.main()
            elif selectedOption == 3:
                listen.main()
            elif selectedOption == 4:
                manageModels.main()
            elif selectedOption == 5:
                userDoesntWantToQuit = False
            else:
                print("There are only 5 options.")
        else:
            print("There are nether negative options nor an option 0.")
    except ValueError:
        print("That was not a number")
