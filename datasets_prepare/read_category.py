import os
import numpy as np
#first column means the number of this class; the second column is the truly counted number of object in evaluation;
numeach_class = np.zeros((12, 7), dtype=np.int32)


def eachFile(filepath):
    pathDir= os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        fopen = open(child, 'r')
        for eachLine in fopen:
            a = eval(eachLine)
            for i in range(12):
                if a[7] == i:
                    numeach_class[i][0] += 1
                    if a[6] == 1:
                        numeach_class[i][1] += 1
                    if a[8] == 0:
                        numeach_class[i][2] += 1
                    if a[8] == 1:
                        numeach_class[i][3] += 1
                    if a[9] == 0:
                        numeach_class[i][4] += 1
                    if a[9] == 1:
                        numeach_class[i][5] += 1
                    if a[9] == 2:
                        numeach_class[i][6] += 1
        fopen.close()
    return numeach_class


if __name__ == '__main__':
    filePathC = "/home/asdf/ECCV2018/ECCV2018/dataset/VisDrone2018-VID-train/annotations/"
    num_class = eachFile(filePathC)
    np.save("/home/asdf/ECCV2018/ECCV2018/dataset/VisDrone2018-VID-train.npy", num_class)

