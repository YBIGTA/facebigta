import torch
from torch import optim
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.autograd import Variable

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data = dsets.ImageFolder('./preprocessed', transform=transform)
loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.cnn = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 64 x 64 x 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 128 x 32 x 32
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 256 x 16 x 16
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 512 x 8 x 8
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            # 512 x 4 x 4
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 30)
        )
    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BatchNorm weight init
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

model = VGG()
model.apply(weights_init)
model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.00002)

for epoch in range(200):
    for i, (images, labels) in enumerate(loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print('Epoch [%d/%d], Iter[%d/%d], Loss: %.4f'
                 %(epoch+1, 200, i+1, len(data)//64, loss.data[0]))


# turn off dropout
model.eval()

correct = 0
total = 0
for images, labels in loader:
    images = Variable(images).cuda()
    outputs = model(images).cuda()
    _, predicted = torch.max(outputs.cpu().data, 1)
    total += labels.cpu().size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the train images: %d %%' % (100 * correct / total))
print("Correct: ", correct)
