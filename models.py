import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

##################################################################

class LeNetMNIST(nn.Module):
    def __init__(self, gain = 1, n_out = 10):
        super(LeNetMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, n_out)
	
        torch.nn.init.xavier_normal_(self.conv1.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.conv2.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain = gain)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LeNetMNIST_train(nn.Module):
    def __init__(self, gain = 1, n_out = 10):
        super(LeNetMNIST_train, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, n_out)
	
        torch.nn.init.xavier_normal_(self.conv1.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.conv2.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain = gain)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
	
##################################################################

class LeNetCIFAR(nn.Module):
    def __init__(self, gain = 1):
        super(LeNetCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

        torch.nn.init.xavier_normal_(self.conv1.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.conv2.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc3.weight, gain = gain)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
	

class LeNetCIFAR_train(nn.Module):
    def __init__(self, gain = 1, n_out = 10):
        super(LeNetCIFAR_train, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, n_out)

        torch.nn.init.xavier_normal_(self.conv1.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.conv2.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc3.weight, gain = gain)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

##################################################################

class LeNetVOC(nn.Module):
    def __init__(self, gain = 1):
        super(LeNetVOC, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.fc1   = nn.Linear(64*111*111, 120)
        self.fc2   = nn.Linear(120, 25)

        torch.nn.init.xavier_normal_(self.conv1.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain = gain)
        torch.nn.init.xavier_normal_(self.fc2.weight, gain = gain)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

##################################################################
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*5*5, 120)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, 10)        # convert matrix with 84 features to a matrix of 10 features (columns)

    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv1(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_1(x)
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv2(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        x = x.view(-1, 16*5*5)
        # FC-1, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)

        return x

##################################################################
class LeNetFilter(nn.Module):
    def __init__(self, kernel, feat, dim, gain=1):
        super(LeNetFilter, self).__init__()
        n_filters = 1

        self.conv1 = nn.Conv2d(dim, n_filters, kernel)
        self.fc1 = nn.Linear(n_filters * feat * feat, 100)

        #self.fc1 = nn.Linear(3*32*32, 10)

        torch.nn.init.xavier_normal_(self.conv1.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.fc1.weight, gain=gain)


    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        #out = x.view(x.size(0), -1)
        #out = self.fc1(out)
        return out

##################################################################

class Net(nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, mid_size)
        self.fc2 = nn.Linear(mid_size, out_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out