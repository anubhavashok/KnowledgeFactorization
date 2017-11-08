from utils import *
from datasets import mnist
from copy import deepcopy
import torch

teacher = torch.load('models/conv4.net')
# Currently using unmodified architecture (10 outputs)
student = deepcopy(teacher)
# Things to try:
# Match logits at output layer (only first 5 outputs of teacher model)
# Match logits at Dropout layer
# Match logits at multiple places

subset = [0, 1, 2, 3, 4]
create_subset(subset, mnist)
trainTeacherStudent(teacher, student, mnist, epochs=10)
