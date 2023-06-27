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
parser.add_argument("-txt", "--text", type=str, default='안녕하세요', help="text to print")
parser.add_argument("-m", "--model", type=str, default='./model/500_trained.pt', help="model checkpoint path")
parser.add_argument("-s", "--source", type=str, default="NanumGothic", help="name of source font")
parser.add_argument("-t", "--target", type=str, default="NanumPen", help="name of target font")
parser.add_argument("-l", "--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("-e", "--epoch", type=int, default=1000, help="number of epochs")
args = parser.parse_args()

test = list(map(lambda x: chr(x), range(44032, 55204)))

test_dataset = FontDataset(args.source, args.target, test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen = torch.load(args.model)
gen.eval()

images = None
for i in args.text:
    for source, gt, emb, name in test_dataset:
        if name == i:
            source = source.to(device)
            emb = emb.to(device)
            made = gen(source, emb)
            if images is None:
                images = made
            else:
                images = torch.cat((images, made), 3)
            break
plt.imshow(images.detach().cpu().reshape(300,-1), cmap='gray')
plt.savefig("./test/"+args.text+".png")



