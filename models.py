import torch
import torch.nn as nn
import torch.nn.functional as F

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
