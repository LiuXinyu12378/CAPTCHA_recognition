import torch
import torchvision
import numpy as np
from PIL import Image
import os
import random

class CAPTCHA(torch.utils.data.Dataset):

    def __init__(self,data_path,train=True):

        self.train = train
        self.root = data_path
        self.data_path = os.listdir(data_path)

        self.target = []
        self.char = []
        self.target_char = []
        for path in self.data_path:
            if path.endswith(".jpg") and path.startswith("target"):
                self.target.append(path)
            elif path.endswith(".jpg") and path.startswith("char"):
                self.char.append(path)

        for target in self.target:
            char = target.replace("target","char")
            if char in self.char:
                self.target_char.append((target,char,1))      #相同

        for target in self.target:
            char = random.choice(self.char)
            if target.replace("target","char") != char:
                self.target_char.append((target,char,0))     #不同

        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize([128,128]),
            torchvision.transforms.ToTensor()
        ])

        random.seed(0.4)
        random.shuffle(self.target_char)
        self.train_set,self.test_set = self.split_list(self.target_char,0.8)

        # print("target:",len(self.target))
        # print("char:",len(self.char))
        # print("target_char:",len(self.target_char))
        print("train_set:",len(self.train_set))
        print("test_set:",len(self.test_set))

    def __getitem__(self,index):
        
        if self.train:
            img1,img2,label = self.train_set[index]
        else:
            img1,img2,label = self.test_set[index]
        img1 = os.path.join(self.root,img1)
        img2 = os.path.join(self.root,img2)
        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(label)
        return (img1,img2),label

    def __len__(self):

        if self.train:
            return(len(self.train_set))
        else:
            return(len(self.test_set))
    
    def split_list(self,lst, ratio):  
        total = len(lst)  
        part1_len = int(total * ratio)  
        part1 = lst[:part1_len]  
        part2 = lst[part1_len:]
        return part1, part2


if __name__ == "__main__":

   dataset =  CAPTCHA("datasets/data_siamese",train=True)

   print(dataset[0])