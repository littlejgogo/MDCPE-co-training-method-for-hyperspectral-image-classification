import numpy as np
from sklearn.metrics import confusion_matrix


def test_data_index(true_label,pred_label,class_num):
	M = 0
	C = np.zeros((class_num+1,class_num+1))
	c1 = confusion_matrix(true_label, pred_label)
	C[0:class_num,0:class_num] = c1
	C[0:class_num,class_num] = np.sum(c1,axis=1)
	C[class_num,0:class_num] = np.sum(c1,axis=0)
	N = np.sum(np.sum(c1,axis=1))
	C[class_num,class_num] = N   # all of the pixel number
	OA = np.trace(C[0:class_num,0:class_num])/N
	every_class = np.zeros((class_num+3,))
	for i in range(class_num):
		acc = C[i,i]/C[i,class_num]
		M = M + C[class_num,i] * C[i,class_num]
		every_class[i] = acc

	kappa = (N * np.trace(C[0:class_num,0:class_num]) - M) / (N*N - M)
	AA = np.sum(every_class,axis=0)/class_num
	every_class[class_num] = OA
	every_class[class_num+1] = AA
	every_class[class_num+2] = kappa
	return every_class, C









def caculate_index(true_label,pred_label,class_num):
	our_label = pred_label
	confusion_matrix = np.zeros((class_num+2,class_num+1))
	for i in range(our_label.shape[0]):
		for j in range(our_label.shape[1]):
			x = our_label[i,j]
			y = true_label[i,j]
			if y==0:
				continue
			confusion_matrix[int(x)-1,int(y)-1] += 1
	confusion_matrix[class_num,:]=np.sum(confusion_matrix[0:class_num,:], axis=0)
	confusion_matrix[0:class_num,class_num]=np.sum(confusion_matrix[0:class_num,0:class_num], axis=1)
	confusion_matrix[class_num+1,0]= confusion_matrix[0,0]/confusion_matrix[class_num,0]
	confusion_matrix[class_num+1,1]= confusion_matrix[1,1]/confusion_matrix[class_num,1]
	confusion_matrix[class_num+1,2]= confusion_matrix[2,2]/confusion_matrix[class_num,2]

	M = 0
	N = our_label.shape[0] * our_label.shape[1] - np.sum(true_label==0)
	for i in range(class_num):
		M = M + confusion_matrix[class_num,i] * confusion_matrix[i,class_num]
	kappa = (N * np.trace(confusion_matrix[0:class_num,0:class_num]) - M) / (N*N - M)
	every_class = confusion_matrix[class_num+1,0:class_num]
	OA = np.trace(confusion_matrix[0:class_num,0:class_num])/N
	AA = np.sum(confusion_matrix[class_num+1,0:class_num])/class_num
	return OA,AA,kappa,every_class

def generate_image(label):
	classifaction_result_img = np.zeros((label.shape[0],label.shape[1], 3))
	a,b = np.where(label==1)
	for location in range(len(a)):
	    classifaction_result_img[a[location],b[location],0] = 255
	    classifaction_result_img[a[location],b[location],1] = 0
	    classifaction_result_img[a[location],b[location],2] = 0
	a,b = np.where(label==2)
	for location in range(len(a)):
	    classifaction_result_img[a[location],b[location],0] = 0
	    classifaction_result_img[a[location],b[location],1] = 255
	    classifaction_result_img[a[location],b[location],2] = 0
	a,b = np.where(label==3)
	for location in range(len(a)):
	    classifaction_result_img[a[location],b[location],0] = 0
	    classifaction_result_img[a[location],b[location],1] = 0
	    classifaction_result_img[a[location],b[location],2] = 255
	
	return classifaction_result_img





