from t_sne_label import *
from model import *
import os
from utils import *
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
from collections.abc import Iterable
from tqdm import tqdm
from data_loader import GetLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
option = 'resnet50'
n_class = 12
batch_size = 32
G = ResBottle(option)

F1 = ResClassifier(num_classes=n_class, num_unit=G.output_num())
F1.apply(weights_init)

G = torch.load('/home/wsco/wb_exper/train_on_img_clef/result/model_G.pth')

G = nn.DataParallel(G).cuda()
F1 = nn.DataParallel(F1).cuda()


contra_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    #GaussianBlur(kernel_size=int(0.1 * 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),

])

test_transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),  # 归一化到[-1.0,1.0]
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
     ])

test_data_source = GetLoader(dataroot="/home/wsco/wb_exper/image_CLEF/p", transforms=test_transform,contra_transforms=contra_transform)
data_loader_s_test = torch.utils.data.DataLoader(test_data_source, batch_size=batch_size, shuffle=False,drop_last=False, num_workers=8)

test_data_target = GetLoader(dataroot="/home/wsco/wb_exper/image_CLEF/c", transforms=test_transform,contra_transforms=contra_transform)
data_loader_t_test = torch.utils.data.DataLoader(test_data_target, batch_size=batch_size, shuffle=False,drop_last=False, num_workers=8)

draw_t_sne(G , F1, data_loader_s_test, data_loader_t_test)