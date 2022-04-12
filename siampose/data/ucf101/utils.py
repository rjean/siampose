import os


def getUCF101ClassMapping(root="datasets/ucf101"):
    mapping = {}
    with open(os.path.join(root, "classInd.txt"), "r") as f:
        for line in f.readlines():
            classId, className = line.strip().split(" ")
            mapping[className] = int(classId)
    assert len(mapping) == 101

    # Define some aliases
    mapping["HandStandPushups"] = mapping["HandstandPushups"]
    mapping["HandStandWalking"] = mapping["HandstandWalking"]

    return mapping
