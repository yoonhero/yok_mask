import json
import pandas as pd

def load_nsmc():
    with open("./tmp/nsmc.txt", "r", encoding="utf-8") as f:
        data = f.read()

    data = data.split("\n")[1:]

    x = []
    y = []
    for d in data:
        temp = d.split("\t")
        index = temp[0]
        label = temp[-1]
        if label not in ["0", "1"]:
            continue
        # document = " ".join(temp[1:-1])
        try:
            document = temp[1]
        except: pass
        x.append(document)
        y.append(label)
    
    return x, y

def load_curse_detection():
    with open("./tmp/curse-detection.txt", "r", encoding="utf-8") as f:
        data = f.read()

    data = data.split("\n")

    x = []
    y = []
    for d in data:
        temp = d.split("|")
        label = temp[-1]
        if label not in ["0", "1"]:
            continue
        document = temp[0] if len(temp) == 2 else "|".join(temp[:-1])
        x.append(document)
        y.append(label)

    return x, y

def load_all():
    x, y = load_nsmc()
    _x, _y = load_curse_detection()
    x = x + _x
    y = y + _y

    return x, y

if __name__ == "__main__":
    load_all()


    
