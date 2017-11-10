from utils import *
from datasets import mnist as dataset 
from copy import deepcopy
from models import conv3
import torch

torch.set_num_threads(4)

teacher = torch.load('models/conv4.net').cuda()
# Currently using unmodified architecture (10 outputs)
student = conv3.CONV3()
#deepcopy(teacher).cuda()
resetModel(student)
# Things to try:
# Match logits at output layer (only first 5 outputs of teacher model)
# Match logits at Dropout layer
# Match logits at multiple places

subset = [9, 8, 7, 6, 5]#[0, 1, 2, 3, 4]
create_subset(subset, dataset)
dataset.net = student.cuda()
for i in range(1, 10):
    dataset.train(i); dataset.test()
#trainTeacherStudent(teacher, student, dataset, epochs=30, lr=0.001)
#trainTeacherStudentIntermediate(teacher, student, dataset, epochs=30, lr=0.001)
