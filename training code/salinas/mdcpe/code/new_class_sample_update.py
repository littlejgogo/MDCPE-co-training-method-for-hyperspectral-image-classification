import numpy as np


# def update1(kmindices1, network_label1, netindices1):
#     jiaojilable1 = network_label1[kmindices1]
#     kmindices1 = np.array(kmindices1)
#     cnn_index1 = {}
#     allcnn_index1 = []
#     for i1 in range(16):
#         indices_eachclass1 = [j1 for j1, x1 in enumerate(jiaojilable1.tolist()) if x1 == i1 + 1]
#         if len(indices_eachclass1) == 0:
#             np.random.shuffle(netindices1)
#             cnn_index1[i1] = netindices1[:8]
#         else:
#             if len(indices_eachclass1) >= 8:
#                 np.random.shuffle(indices_eachclass1)
#                 aaa1 = indices_eachclass1[:8]
#                 bbb1 = kmindices1[aaa1]
#                 cnn_index1[i1] = bbb1.tolist()
#             else:
#                 if len(indices_eachclass1) <= 3:
#                     np.random.shuffle(indices_eachclass1)
#                     indices_eachclass1 = np.array(indices_eachclass1)
#                     aaa1 = np.repeat(indices_eachclass1, 3, axis=0)
#                     bbb1 = kmindices1[aaa1.tolist()]
#                     cnn_index1[i1] = bbb1.tolist()
#                 else:
#                     np.random.shuffle(indices_eachclass1)
#                     indices_eachclass1 = np.array(indices_eachclass1)
#                     aaa1 = np.repeat(indices_eachclass1, 2, axis=0)
#                     bbb1 = kmindices1[aaa1.tolist()]
#                     cnn_index1[i1] = bbb1.tolist()
#         allcnn_index1 += cnn_index1[i1]
#     np.random.shuffle(allcnn_index1)
#
#     return allcnn_index1
#
#
# def update2(kmindices, network_label, netindices):
#     jiaojilable = network_label[kmindices]
#     kmindices = np.array(kmindices)
#     cnn_index = {}
#     allcnn_index = []
#     for i in range(16):
#         indices_eachclass = [j for j, x in enumerate(jiaojilable.tolist()) if x == i + 1]
#         if len(indices_eachclass) == 0:
#             np.random.shuffle(netindices)
#             cnn_index[i] = netindices[:15]
#         else:
#             if len(indices_eachclass) >= 15:
#                 np.random.shuffle(indices_eachclass)
#                 aaa = indices_eachclass[:15]
#                 bbb = kmindices[aaa]
#                 cnn_index[i] = bbb.tolist()
#             else:
#                 if len(indices_eachclass) <= 4:
#                     np.random.shuffle(indices_eachclass)
#                     indices_eachclass = np.array(indices_eachclass)
#                     aaa = np.repeat(indices_eachclass, 4, axis=0)
#                     bbb = kmindices[aaa.tolist()]
#                     cnn_index[i] = bbb.tolist()
#                 else:
#                     if len(indices_eachclass) < 10:
#                         np.random.shuffle(indices_eachclass)
#                         indices_eachclass = np.array(indices_eachclass)
#                         aaa = np.repeat(indices_eachclass, 2, axis=0)
#                         bbb = kmindices[aaa.tolist()]
#                         cnn_index[i] = bbb.tolist()
#                     else:
#                         np.random.shuffle(indices_eachclass)
#                         indices_eachclass = np.array(indices_eachclass)
#                         aaa = indices_eachclass
#                         bbb = kmindices[aaa.tolist()]
#                         cnn_index[i] = bbb.tolist()
#         allcnn_index += cnn_index[i]
#     np.random.shuffle(allcnn_index)
#
#     return allcnn_index

def update1(kmindices1, network_label1, netindices1):
    jiaojilable1 = network_label1[kmindices1]
    kmindices1 = np.array(kmindices1)
    cnn_index1 = {}
    allcnn_index1 = []
    for i1 in range(16):
        indices_eachclass1 = [j1 for j1, x1 in enumerate(jiaojilable1.tolist()) if x1 == i1 + 1]
        if len(indices_eachclass1) == 0:
            np.random.shuffle(netindices1)
            cnn_index1[i1] = netindices1[:7]
        else:
            if len(indices_eachclass1) >= 7:
                np.random.shuffle(indices_eachclass1)
                aaa1 = indices_eachclass1[:7]
                bbb1 = kmindices1[aaa1]
                cnn_index1[i1] = bbb1.tolist()
            else:
                if len(indices_eachclass1) >= 5:
                    np.random.shuffle(indices_eachclass1)
                    indices_eachclass1 = np.array(indices_eachclass1)
                    aaa1 = indices_eachclass1
                    bbb1 = kmindices1[aaa1.tolist()]
                    cnn_index1[i1] = bbb1.tolist()
                else:
                    if len(indices_eachclass1) >= 3:
                        np.random.shuffle(indices_eachclass1)
                        indices_eachclass1 = np.array(indices_eachclass1)
                        aaa1 = np.repeat(indices_eachclass1, 2, axis=0)
                        bbb1 = kmindices1[aaa1.tolist()]
                        cnn_index1[i1] = bbb1.tolist()
                    else:
                        np.random.shuffle(indices_eachclass1)
                        indices_eachclass1 = np.array(indices_eachclass1)
                        aaa1 = np.repeat(indices_eachclass1, 5, axis=0)
                        bbb1 = kmindices1[aaa1.tolist()]
                        cnn_index1[i1] = bbb1.tolist()
        allcnn_index1 += cnn_index1[i1]
    np.random.shuffle(allcnn_index1)

    return allcnn_index1


def update2(kmindices, network_label, netindices):
    jiaojilable = network_label[kmindices]
    kmindices = np.array(kmindices)
    cnn_index = {}
    allcnn_index = []
    for i in range(16):
        indices_eachclass = [j for j, x in enumerate(jiaojilable.tolist()) if x == i + 1]
        if len(indices_eachclass) == 0:
            np.random.shuffle(netindices)
            cnn_index[i] = netindices[:7]
        else:
            if len(indices_eachclass) >= 7:
                np.random.shuffle(indices_eachclass)
                aaa = indices_eachclass[:7]
                bbb = kmindices[aaa]
                cnn_index[i] = bbb.tolist()
            else:
                if len(indices_eachclass) >= 5:
                    np.random.shuffle(indices_eachclass)
                    indices_eachclass = np.array(indices_eachclass)
                    aaa = indices_eachclass
                    bbb = kmindices[aaa.tolist()]
                    cnn_index[i] = bbb.tolist()
                else:
                    if len(indices_eachclass) >= 3:
                        np.random.shuffle(indices_eachclass)
                        indices_eachclass = np.array(indices_eachclass)
                        aaa = np.repeat(indices_eachclass, 2, axis=0)
                        bbb = kmindices[aaa.tolist()]
                        cnn_index[i] = bbb.tolist()
                    else:
                        np.random.shuffle(indices_eachclass)
                        indices_eachclass = np.array(indices_eachclass)
                        aaa = np.repeat(indices_eachclass, 5, axis=0)
                        bbb = kmindices[aaa.tolist()]
                        cnn_index[i] = bbb.tolist()
        allcnn_index += cnn_index[i]
    np.random.shuffle(allcnn_index)

    return allcnn_index