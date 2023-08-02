
# Set up data loaders
from dataset3 import CAPTCHA
from networks import EmbeddingNet, ClassificationNet
from metrics import AccumulatedAccuracyMetric,AverageNonzeroTripletsMetric
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

from trainer import fit,mps,device
import numpy as np
cuda = torch.cuda.is_available()


#import matplotlib
#import matplotlib.pyplot as plt

path = "datasets/data_siamese"

siamese_train_dataset = CAPTCHA(path,train=True) # Returns pairs of images and target same/different
siamese_test_dataset = CAPTCHA(path,train=False)
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, SiameseNet,TripletNet
from losses import ContrastiveLoss,TripletLoss,OnlineTripletLoss

margin = 3.0
embedding_net = EmbeddingNet()
# model = SiameseNet(embedding_net)
model = TripletNet(embedding_net)
if cuda:
    model.cuda()

if mps:
    model.to(device)
# loss_fn = ContrastiveLoss(margin)

loss_fn = TripletLoss(margin)
#loss_fn = OnlineTripletLoss(margin)


lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 100


fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)



