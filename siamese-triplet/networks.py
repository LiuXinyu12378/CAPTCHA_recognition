import torch.nn as nn
import torch.nn.functional as F
import torchvision

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(10816, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )
        self.resnet18_1 =torchvision.models.resnet18(pretrained=True)
        self.resnet18_2 =torchvision.models.resnet18(pretrained=True)

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        # output = self.resnet18_1(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        # self.embedding_net = embedding_net
        self.resnet18_1 =torchvision.models.resnet18(pretrained=True)
        self.resnet18_2 =torchvision.models.resnet18(pretrained=True)

    def forward(self, x1, x2):
        # output1 = self.embedding_net(x1)
        # output2 = self.embedding_net(x2)
        output1 = self.resnet18_1(x1)
        output2 = self.resnet18_1(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        self.embedding_net_1 = EmbeddingNet()
        self.embedding_net_2 = EmbeddingNet()
        self.resnet18_1 =torchvision.models.resnet18(pretrained=True)
        self.resnet18_2 =torchvision.models.resnet18(pretrained=True)
        self.resnet34_1 =torchvision.models.resnet34(pretrained=True)
        self.resnet34_2 =torchvision.models.resnet34(pretrained=True)
        self.resnet50_1 =torchvision.models.resnet50(pretrained=True)
        self.resnet50_2 =torchvision.models.resnet50(pretrained=True)


    def forward(self, x1, x2, x3):
        output1 = self.resnet18_1(x1)
        output2 = self.resnet18_2(x2)
        output3 = self.resnet18_2(x3)
        return output1, output2, output3
    
    def forward_x1(self, x1):
        output1 = self.resnet18_1(x1)
        return output1
    
    def forward_x2(self, x2):
        output2 = self.resnet18_2(x2)
        return output2 

    def get_embedding(self, x):
        return self.embedding_net(x)
