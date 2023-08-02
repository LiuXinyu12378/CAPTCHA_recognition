
from networks import EmbeddingNet, SiameseNet,TripletNet
from losses import ContrastiveLoss,TripletLoss,OnlineTripletLoss
import torch
from PIL import Image
import torchvision
import torch.nn.functional as F

cuda = torch.cuda.is_available()

embedding_net = EmbeddingNet()
# model = SiameseNet(embedding_net)
model = TripletNet(embedding_net)

model.load_state_dict(torch.load('siamese-triplet/tripletnet_resnet18_12.pth',map_location=torch.device('cpu')))

if cuda:
    model.cuda()

transform = torchvision.transforms.Compose([
    # torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Resize([64,64]),
    torchvision.transforms.ToTensor()
])


if __name__ == "__main__":
    x1 = "/Users/xinyuuliu/Desktop/验证码检测/datasets/data_siamese/char_u4e01_img_3318.jpg"
    x2 = "/Users/xinyuuliu/Desktop/验证码检测/datasets/data_siamese/char_u9c9c_img_3423.jpg"
    x3 = "/Users/xinyuuliu/Desktop/验证码检测/datasets/data_siamese/char_u9e2d_img_3470.jpg"
    
    x4 = "/Users/xinyuuliu/Desktop/验证码检测/datasets/data_siamese/target_u4e01_img_3318.jpg"
    x5 = "/Users/xinyuuliu/Desktop/验证码检测/datasets/data_siamese/target_u9c9c_img_3423.jpg"
    x6 = "/Users/xinyuuliu/Desktop/验证码检测/datasets/data_siamese/target_u9e2d_img_3470.jpg"
    x7 = "/Users/xinyuuliu/Desktop/验证码检测/datasets/data_siamese/target_u7cd5_img_3368.jpg"
    x8 = "/Users/xinyuuliu/Desktop/验证码检测/datasets/data_siamese/target_u7cd6_img_3305.jpg"

    x1s = []
    x2s = []
    for i in [x1,x2,x3]:
        img = Image.open(i).convert('RGB')
        img = transform(img).unsqueeze_(0)
        x1s.append(img)
    
    for i in [x4,x5,x6,x7,x8]:
        img = Image.open(i).convert('RGB')
        img = transform(img).unsqueeze_(0)
        x2s.append(img)

    x_1 = torch.cat(x1s,dim=0)
    x_2 = torch.cat(x2s,dim=0)

    output1 = model.forward_x1(x_1)
    output2 = model.forward_x2(x_2)

    for x1 in output1:
        distance = F.pairwise_distance(x1,output2)
        print(distance,torch.argmin(distance))
    




