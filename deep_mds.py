from torch import nn
import torch
from torch.autograd import Variable
import torch.optim as optim
import time
import argparse
from utils import removeLayers
from models import alexnet, conv4, resnet18
from models.finetune import FinetuneModel
import torch.nn.functional as F
from utils import resetModel, create_subset

#TODO: Check if the symmetric loss is better or worse than the the siamese style training using pairs


def test_dataset(student, dataset):

    student.add_module('LogSoftmax', nn.LogSoftmax())
    dataset.net = student
    removeLayers(student, type='LogSoftmax')
    accuracy = dataset.test()
    return accuracy


def get_distance_matrix(feats):

    # tested this
    # normalized fc7 distance for stability
    norm = feats.norm(p=2, dim=1, keepdim=True)
    feats = feats.div(norm.expand_as(feats))

    # https: // discuss.pytorch.org / t / efficient - distance - matrix - computation / 9065 / 3
    xx = torch.mul(feats, feats)
    xx = torch.sum(xx, dim=1)
    batch_size = feats.size()[0]

    xx = xx.expand(batch_size, batch_size)
    xy = -2*torch.matmul(feats, feats.t())
    dist_mat = xx + xy + xx.t()

    return dist_mat


def train_teacher_student_mds(teacher, student, dataset, epochs=5):

    if args.cuda:
        teacher = teacher.cuda()
        student = student.cuda()

    startTime = time.time()
    optimizer = optim.SGD(student.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          nesterov=True,
                          weight_decay=5e-4)

    for i in range(1, epochs+1):
        for b_idx, (data, target) in enumerate(dataset.train_loader):

            losses = []
            student.train()

            if args.cuda:
                data = data.cuda()

            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()

            student_feats = student.extract_features(data)
            teacher_feats = (teacher.extract_features(data)).detach()

            student_output = get_distance_matrix(student_feats)
            teacher_output = get_distance_matrix(teacher_feats)
            diff = (student_output - teacher_output)
            diff = torch.mul(diff, diff)
            diff = torch.sum(diff)

            batch_size = student_feats.size()[0]
            pair_loss = (1.0/batch_size)*0.5*diff

            losses.append(args.lambda_constant*pair_loss)

            if args.ce_loss:
                predicted = student.classifier(student_feats)
                ce_loss = F.nll_loss(F.log_softmax(predicted), target)
                losses.append(ce_loss)

            # if args.kd_loss:
            # TODO: Add KD loss and append to losses

            loss = sum(losses)
            loss.backward()
            optimizer.step()

            print('Loss: {:.6f}'.format(loss.data[0]))

        accuracy = test_dataset(student, dataset)
        print('Train Epoch: {} \tLoss: {:.6f} \tAccuracy: {:.6f}'.format(i, loss.data[0], accuracy))

    accuracy = test_dataset(student, dataset)
    print('Final accuracy: {:.6f}'.format(accuracy))
    print('Time elapsed: {}'.format(time.time()-startTime))
    return accuracy


def finetune_teacher(teacher, dataset):

    # this just works with resnet. confirm with bhav
    teacher.maxpool = nn.Dropout(0)
    teacher.avgpool = nn.AvgPool2d(2, 2)
    teacher.fc = nn.Linear(512, args.num_outputs)

    for p in teacher.pre_layers.parameters():
        p.requires_grad = False

    for p in teacher.layer1.parameters():
        p.requires_grad = False

    for p in teacher.layer2.parameters():
        p.requires_grad = False

    for p in teacher.layer3.parameters():
        p.requires_grad = False

    for p in teacher.layer4.parameters():
        p.requires_grad = False

    for p in teacher.linear.parameters():
        p.requires_grad = True

    dataset.net = teacher

    if args.cuda:
        teacher = teacher.cuda()

    for i in range(1, 20):
        dataset.train(i)
        dataset.test()

    return teacher


def get_dataset(dataset_name):

    if dataset_name == 'cifar10':
        from datasets import cifar10 as dataset
    if dataset_name == 'mnist':
        from datasets import mnist as dataset

    subset = range(args.num_outputs)
    create_subset(subset, dataset)
    return dataset


def main():

    dataset = get_dataset(args.dataset)
    teacher = resnet18.resnet18()

    if args.finetune:
        teacher = finetune_teacher(teacher, dataset)

    student = conv4.CONV4(num_outputs=args.num_outputs, num_channels=3)
    train_teacher_student_mds(teacher, student, dataset, epochs=args.epochs)

    # This is unsupervised training using just pairwise distances
    if args.additional_training:
        add_dataset = get_dataset(args.add_dataset)
        train_teacher_student_mds(teacher, student, add_dataset, epochs=args.add_epochs)


if __name__ == '__main__':

    # Train settings
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--path', type=str, default='', help='path to pretrained model')
    parser.add_argument('--log_path', type=str, default='', help='path to store logs')

    parser.add_argument('--dataset', type=str, default='cifar10', help='Which data set to test')
    parser.add_argument('--num_outputs', type=int, default=10, help='number of outputs to predict')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--inference', type=int, default=False, help='Whether to run training loop')
    parser.add_argument('--scratch', type=int, default=True, help='train from scratch')

    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')

    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1000, help='how many batches to wait before running on validation set')

    parser.add_argument('--ce_loss', action='store_true', default=True, help='Use Cross Entropy loss')
    parser.add_argument('--kd_loss', action='store_true', default=False, help='Use KD loss')
    parser.add_argument('--lambda_constant', type=float, default=0.1, help='Lambda value to balance the two losses')

    parser.add_argument('--add_training', action='store_true', default=False, help='Unsupervised learning using only paired loss')
    parser.add_argument('--add_dataset', type=str, default=0.1, help='Additional dataset to train on')
    parser.add_argument('--add_epochs', type=int, default=10, help='Number of epochs to train on')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main()
