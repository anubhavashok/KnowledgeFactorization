from utils import *
from datasets import mnist
from copy import deepcopy
import torch

torch.set_num_threads(4)

teacher = torch.load('models/conv4.net').cuda()
# Currently using unmodified architecture (10 outputs)
student = deepcopy(teacher).cuda()
# Things to try:
# Match logits at output layer (only first 5 outputs of teacher model)
# Match logits at Dropout layer
# Match logits at multiple places

subset = [1]
create_subset(subset, mnist)
mnist.net = student.cuda()
#for i in range(1, 10):
#    mnist.train(i); mnist.test()
trainTeacherStudent(teacher, student, mnist, epochs=10, lr=0.0005)
