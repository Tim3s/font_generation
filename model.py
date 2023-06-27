import torch.nn as nn
import torch

class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 1, 5, 1, 2),
            nn.Sigmoid()
        )
        self.emb = nn.Embedding(184, embedding_dim=90000)
    
    def forward(self, s, emb):
        s = s.reshape(-1,1,300,300)
        emb = self.emb(emb)
        emb = emb.reshape(-1, 3, 300, 300)
        s = torch.cat((emb,s),1)
        
        return self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(s)))))
    
    
class Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4,8,5,1,2),
            nn.MaxPool2d(5),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8,16,5,1,2),
            nn.MaxPool2d(3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400,1000),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(1000,1),
            nn.Sigmoid()
        )
        self.emb = nn.Embedding(184, embedding_dim=90000)
    
    def forward(self, img, emb):
        emb = self.emb(emb)
        emb = emb.reshape(-1, 3, 300, 300)
        img = torch.cat((emb, img), 1)
        return self.linear2(self.linear1(self.conv2(self.conv1(img))))