from torchvision import models
from utils import *
from datasets import cifar10

teacher = models.resnet34(pretrained=True).cuda()
student = models.resnet18(pretrained=True).cuda()

trainTeacherStudent(teacher, student, cifar10, epochs=10)
