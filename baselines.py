import torch
from torchvision import models, datasets, transforms
import argparse
from utils import *
# from datasets import mnist
from models.conv4 import CONV4, FinetuneModel
from models.vgg import VGG
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import namedtuple


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--path', type=str, default='', metavar='N', help='path to pretrained model')
parser.add_argument('--log_path', type=str, default='', metavar='N', help='path to store logs')
parser.add_argument('--pretrained', type=str, default='', metavar='N', help='path to load pretrained model from')

parser.add_argument('--dataset', type=str, default='mnist', help='Which data set to test')
parser.add_argument('--num_outputs', type=int, default=10, help='number of outputs to predict')

parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--inference', type=int, default=False, help='Whether to run training loop')
parser.add_argument('--scratch', type=int, default=True, help='train from scratch')

parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--batch_size', type=int, default=10, metavar='N', help='batch size during training')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')

parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1000, metavar='N', help='how many batches to wait before running on validation set')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print args


optimizer = None
f = None
my_dataset = None


def train(net, epoch):
    global f, optimizer
    if epoch == 1:

        # define your optimizer
        if args.dataset == 'mnist':
            # optimizer = optim.Adam(net.parameters(), lr=args.lr)
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001)
        elif args.dataset == 'cifar10':
            print 'Here cifar10'
            # optimizer = optim.Adam( filter(lambda p: p.requires_grad, net.parameters()), lr=0.005, weight_decay=1e-4)
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.005)

        f = open(args.log_path, 'w')

    avg_loss = 0

    for batch_idx, (data, target) in enumerate(my_dataset.train_loader):
        net.train()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net.forward(data)
        loss = F.nll_loss(F.log_softmax(output), target)
        loss.backward()
        optimizer.step()

        avg_loss += loss.data[0]

        if batch_idx % args.log_interval == 0:
            avg_loss /= args.log_interval
            log_string = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(my_dataset.train_loader) * args.batch_size,
                100. * batch_idx / (len(my_dataset.train_loader) * args.batch_size), avg_loss)

            print(log_string)
            f.write(log_string)
            avg_loss = 0

        if batch_idx % args.test_interval == 0:
            test(net)


def test(net):
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in my_dataset.test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net.forward(data)
        test_loss += F.nll_loss(F.log_softmax(output), target).data[0]
        pred = output.data.max(1)[1]    # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(my_dataset.test_loader)   # loss function already averages over batch size
    acc = float(correct) / (len(my_dataset.test_loader)*args.batch_size)


    log_string = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(my_dataset.test_loader)*args.batch_size,
        100. * correct / (len(my_dataset.test_loader)*args.batch_size))

    print(log_string)
    if not args.inference and f:
        f.write(log_string)

    return acc


def get_dataset():

    my_dataset = namedtuple('my_dataset', ['train_loader', 'test_loader', 'args', 'kwargs'])
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    # initialize loaders
    my_dataset.kwargs = kwargs

    if args.dataset == 'mnist':
        my_dataset.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        my_dataset.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar10':
        my_dataset.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        my_dataset.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    my_dataset.args = args
    return my_dataset



def main():

    # create subset
    global my_dataset, optimizer, f

    my_dataset = get_dataset()
    subset = []
    for i in xrange(args.num_outputs):
        subset.append(i)
    create_subset(subset, my_dataset)

    # define your model
    if args.scratch:
        model = CONV4(args.num_outputs, 1)
        # model = VGG('VGG19')
        resetModel(model)
    else:
        model = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        removeLayers(model, 'LogSoftmax')
        model = FinetuneModel(model, 'conv4', args.num_outputs, True)

    # Only inference
    if args.inference:
        model = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        final_accuracy = test(model)
    else:
        print model

        # Train / test loop
        for i in range(args.epochs):
            print "Epoch: ", i
            train(model, i+1)
            model_name = 'conv4_{0}_{1}.net'.format(args.num_outputs, i+1)
            torch.save(model, args.path+model_name)
        if f:
            f.close()


main()


# run at four settings
# 1. Train for conv4 10
# 2. Train for conv4 5
# 3. Finetune for conv4 5
# 4. Predict for 5 using conv4 10

# path ./models/conv4_10.net
# path ./models/conv4_5.net
# path ./models/conv4_10_conv4_5.net






