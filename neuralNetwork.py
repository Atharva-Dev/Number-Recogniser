import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size = 5)
        self.conv2 = nn.Conv2d(10,30,kernel_size = 5)
        self.conv2_drop = nn.Dropout2d() 
        self.fc1 = nn.Linear(480, 200)
        self.fc2 = nn.Linear(200,300)
        self.fc3 = nn.Linear(300, 50)
        self.out = nn.Linear(50, 10)

    def forward(self, x):
        
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(x)
        
        x = F.dropout(x, training=True)
        x = x.view(-1, 480)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.out(x)
        
        x = F.log_softmax(x)
        
        return x


if __name__ == "__main__":
    trainLoader = DataLoader(torchvision.datasets.MNIST('./mnist_', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])), batch_size=60, shuffle=True)
    
    network = NeuralNet()
    network.load_state_dict(torch.load('./brain.pth'))
    optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)

    for _ in range(50):
        network.train()
        for index, (data, target) in enumerate(trainLoader):
            
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if index % 25 == 0:
                print('loss = ',loss.item())

    torch.save(network.state_dict(),"./brain.pth")
        
        
    

        



