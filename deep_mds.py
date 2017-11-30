from torch import nn
import torch
from torchvision import models, datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
import time
import argparse
from utils import removeLayers
from models import alexnet, conv4

#TODO: Add option to use CELoss as well for when labels are available (First loss is unsupervised, second term is for supervised)
#TODO: Check if the symmetric loss is better or worse than the the siamese style training using pairs


def test_dataset(student, dataset):

    student.add_module('LogSoftmax', nn.LogSoftmax())
    dataset.net = student
    removeLayers(student, type='LogSoftmax')
    accuracy = dataset.test()
    return accuracy


def get_distance_matrix(feats):

    xx = 2*torch.mul(feats, feats)
    xy = -2*torch.mat_mul(feats, feats.t())
    xx = torch.sum(xx, dim=1)
    dist_mat = xx + xy

    return dist_mat


def train_teacher_student_mds(teacher, student, dataset, epochs=5):

    if args.cuda:
        teacher = teacher.cuda()
        student = student.cuda()

    startTime = time.time()

    MSEloss = nn.MSELoss().cuda()
    optimizer = optim.SGD(student.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          nesterov=True,
                          weight_decay=5e-4)

    student.train()
    for i in range(1, epochs+1):
        for b_idx, (data, targets) in enumerate(dataset.train_loader):
            data = data.cuda()
            data = Variable(data)
            optimizer.zero_grad()

            student_feats = student.extract_features(data)
            teacher_feats = (teacher.extract_features(data)).detach()

            student_output = get_distance_matrix(student_feats)
            teacher_output = get_distance_matrix(teacher_feats)

            loss = MSEloss(student_output, teacher_output)
            loss = 0.5*loss

            loss.backward()
            optimizer.step()

        accuracy = test_dataset(student, dataset)
        print('Train Epoch: {} \tLoss: {:.6f} \tAccuracy: {:.6f}'.format(i, loss.data[0], accuracy))

    accuracy = test_dataset(student, dataset)
    print('Final accuracy: {:.6f}'.format(accuracy))
    print('Time elapsed: {}'.format(time.time()-startTime))
    return accuracy


def main():

    if args.dataset == 'cifar10':
        from datasets import cifar10 as dataset
    if args.dataset == 'mnist':
        from datasets import cifar10 as dataset


    teacher = alexnet.AlexNet(pretrained=True)
    student = conv4.CONV4()

    train_teacher_student_mds(teacher, student, dataset, epochs=args.epochs, lr=args.lr)


if __name__ == '__main__':

    # Train settings
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--path', type=str, default='', metavar='N', help='path to pretrained model')
    parser.add_argument('--log_path', type=str, default='', metavar='N', help='path to store logs')

    parser.add_argument('--dataset', type=str, default='mnist', help='Which data set to test')
    parser.add_argument('--num_outputs', type=int, default=10, help='number of outputs to predict')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--inference', type=int, default=False, help='Whether to run training loop')
    parser.add_argument('--scratch', type=int, default=True, help='train from scratch')

    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')

    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before running on validation set')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main()
