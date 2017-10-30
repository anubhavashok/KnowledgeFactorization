from torchvision import models
from utils import *
from datasets import cifar10
from torch import nn

teacher = models.resnet34(pretrained=True)
# modify model to run on cifar10
teacher.maxpool = nn.Dropout(0)
teacher.avgpool = nn.AvgPool2d(2, 2)
teacher.fc = nn.Linear(512, 10)
cifar10.net = teacher
teacher = teacher.cuda()
for i in range(1, 10):
    cifar10.train(i)
    cifar10.test()

student = models.resnet18(pretrained=True)
student.maxpool = nn.Dropout(0)
student.avgpool = nn.AvgPool2d(2, 2)
student.fc = nn.Linear(512, 10)
student = student.cuda()

trainTeacherStudent(teacher, student, cifar10, epochs=10)
