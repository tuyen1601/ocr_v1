import os

def Init(folder1,folder2,folder3):
    try:
        for file in os.listdir(folder1):
            os.remove(folder1+file)
    except:
        pass
    try:
        for file in os.listdir(folder2):
            os.remove(folder2+file)
    except:
        pass
    try:
        for file in os.listdir(folder3):
            os.remove(folder3+file)
    except:
        pass