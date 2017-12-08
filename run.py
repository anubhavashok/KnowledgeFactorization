'''
Currently modified for the Imagenet32 -> Stanford Dogs experiment
'''

from torchvision import models
from utils import *
from datasets import stanfordDogs
from torch import nn
import torch

teacher = models.resnet34(pretrained=True)
# modify model to run on cifar10
#teacher.maxpool = nn.Dropout(0)
#teacher.avgpool = nn.AvgPool2d(2, 2)
#teacher.fc = nn.Linear(512, 10)
#imagenet32.net = teacher
#teacher = teacher.cuda()
teacher = torch.nn.DataParallel(teacher,device_ids=[0,1,2,3])
'''
for i in range(1, 10):
    cifar10.train(i)
    cifar10.test()
'''
student = models.resnet18()
#student.maxpool = nn.Dropout(0)
#student.avgpool = nn.AvgPool2d(2, 2)
#student.fc = nn.Linear(512, 10)
#student = student.cuda()
student = torch.nn.DataParallel(student,device_ids=[0,1,2,3])

trainTeacherStudent(teacher, student, stanfordDogs, epochs=10, lr = 0.0005)
torch.save(student.cpu(), 'imagenet_to_dogs.net')
