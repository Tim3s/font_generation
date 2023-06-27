import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import FontDataset
from torch.utils.data import DataLoader
from model import Gen, Dis
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='train model')
parser.add_argument("-s", "--source", type=str, default="NanumGothic", help="name of source font")
parser.add_argument("-t", "--target", type=str, default="NanumPen", help="name of target font")
parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch size")
parser.add_argument("-l", "--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("-e", "--epoch", type=int, default=1000, help="number of epochs")
args = parser.parse_args()

train = list(map(lambda x: chr(x), range(44032, 55204, 25)))
val = list(map(lambda x: chr(x), range(44062, 55204, 25)))

train_dataset = FontDataset(args.source, args.target, train)
val_dataset = FontDataset(args.source, args.target, val)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen = Gen().to(device)
dis = Dis().to(device)

criterion = nn.BCELoss().to(device)
G_optimizer = optim.Adam(gen.parameters(), lr=args.learning_rate)
D_optimizer = optim.Adam(dis.parameters(), lr=args.learning_rate)

genloss = []
disloss = []

gen.train()
dis.train()

for epoch in range(args.epoch):
    T = time.time()
    
    for s, t, emb, name in train_loader:

        s = s.to(device)
        t = t.to(device)
        emb = emb.to(device)
        real_label = torch.ones(s.shape[0], 1).type(torch.float).to(device)
        fake_label = torch.zeros(t.shape[0], 1).type(torch.float).to(device)
        
        for _ in range(1):
            D_optimizer.zero_grad()
            
            fake_img = gen(s, emb)

            real = dis(t, emb)
            real_loss = criterion(real, real_label)
            
            fake = dis(fake_img.detach(), emb)
            fake_loss = criterion(fake, fake_label)
            loss = (real_loss + fake_loss) / 2
            loss.backward()
            D_optimizer.step()
            
        disloss.append(loss.item())
        
        G_optimizer.zero_grad()
        
        fake_label = torch.ones(s.shape[0], 1).type(torch.float).to(device)
        
        fake_img = gen(s, emb)

        fake = dis(fake_img, emb)
        loss = criterion(fake, real_label)
        loss.backward()
        G_optimizer.step()
        genloss.append(loss.item())
    
    
    print(f'[Epoch: {epoch+1:0>4}] Discriminative loss: {disloss[-1]:.4f}   ', end='')
    print(f'Generative loss: {genloss[-1]:.4f}   time: {time.time()-T:.1f}s')
    plt.imshow(fake_img[0].detach().cpu().reshape(300,300), cmap='gray')
    plt.savefig(f"./figs/epoch{epoch+1:0>4}_{name[0]}.png")
    plt.imshow(t[0].detach().cpu().reshape(300,300), cmap='gray')
    plt.savefig(f"./figs/epoch{epoch+1:0>4}_{name[0]}_gt.png")
    if epoch % 5 == 4 or epoch == args.epoch-1:
        torch.save(gen, f"./model/{epoch+1}_trained.pt")