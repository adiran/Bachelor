import os

#own imports
import functions as f
import config as conf

#manages all models that have been computed
def main():
    userWantNotToQuit = True
    models = f.loadModels()
    while userWantNotToQuit:
        print("Currently stored models:")
        for i in range(len(models)):
            print("Nr. " +
                 str(i +
                     1) +
                 ": " +
                 str(models[i].name) +
                 " | Frames:\t" +
                 str(len(models[i].features)) +
                 " | Matches:\t" +
                 str(models[i].matches) +
                 " | Influenced by:\t" +
                 str(models[i].influencedBy) +
                 " | Tolerance:\t" +
                 str(models[i].tolerance) +
                 " | Score:\t" +
                 str(models[i].score) +
                 " | Loaded:\t" +
                 str(models[i].loaded))
        print()
        printCommands()
        isUserInputWrong = True
        while isUserInputWrong:
            userInput = raw_input("What do you want to do? ")
            if userInput == "a":
                userInput = raw_input("Which model do you want to activate? ")
                try:
                    selectedModel = int(userInput) - 1
                    if selectedModel < len(models):
                        if selectedModel >= 0:
                            models[selectedModel].activate()
                            f.storeSameModel(models[selectedModel])
                            isUserInputWrong = False
                        else:
                            print("There are no models with a number smaller or equal 0.")
                    else:
                        print("There is no model with such a high number.")
                except ValueError:
                    print("That was not a number")
            elif userInput == "d":
                userInput = raw_input("Which model do you want to deactivate? ")
                try:
                    selectedModel = int(userInput) - 1
                    if selectedModel < len(models):
                        if selectedModel >= 0:
                            models[selectedModel].deactivate()
                            f.storeSameModel(models[selectedModel])
                            isUserInputWrong = False
                        else:
                            print("There are no models with a number smaller or equal 0.")
                    else:
                        print("There is no model with such a high number.")
                except ValueError:
                    print("That was not a number")
            elif userInput == "del":
                userInput = raw_input("Which model do you want to delete? ")
                try:
                    selectedModel = int(userInput) - 1
                    if selectedModel < len(models):
                        if selectedModel >= 0:
                            os.remove(conf.MODELS_DIR + "/" + models[selectedModel].name)
                            models = f.loadModels()
                            isUserInputWrong = False
                        else:
                            print("There are no models with a number smaller or equal 0.")
                    else:
                        print("There is no model with such a high number.")
                except ValueError:
                    print("That was not a number")
            elif userInput == "r":
                userInput = raw_input("Which model do you want to rename? ")
                try:
                    selectedModel = int(userInput) - 1
                    if selectedModel < len(models):
                        if selectedModel >= 0:
                            oldModelName = models[selectedModel].name
                            newName = raw_input("Which name should this model have?")
                            models[selectedModel].name = newName
                            f.storeModel(models[selectedModel])
                            os.remove(conf.MODELS_DIR + "/" + oldModelName)
                            isUserInputWrong = False
                        else:
                            print("There are no models with a number smaller or equal 0.")
                    else:
                        print("There is no model with such a high number.")
                except ValueError:
                    print("That was not a number")
            elif userInput == "h":
                isUserInputWrong = False
                printCommands()
            elif userInput == "q":
                isUserInputWrong = False
                userWantNotToQuit = False
            else:
                print("That was not a valid input.")

#prints all possible commands
def printCommands():
    print("There are the following commands:")
    print("\ta\t(a)ctivate a model")
    print("\td\t(d)eactivate a model")
    print("\tdel\t(del)ete a model")
    print("\tr\t(r)ename a model")
    print("\th\tprint this (h)elp again")
    print("\tq\t(q)uit the model management")
