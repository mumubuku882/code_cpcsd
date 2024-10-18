import os
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
from collections.abc import Iterable
from tqdm import tqdm
from data_loader import GetLoader
from model import *
from loss import *
import os
from utils import *
from torch.autograd import Variable
import torch.nn.functional as F
from t_sne import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#一些参数初始化
data_folder = '/home/ubuntu/zm/my/image_CLEF'
batch_size = 32
n_class = 12
n_epoch = 200

#num_k = 4
criterion_cross = Cross_entropy_loss()
criterion_contra = InfoNCE_loss()
neg = Neg_MI()

option = 'resnet50'
#num_layer = 2 #分类器层数
G = ResBottle(option)
F1 = ResClassifier(num_classes=n_class, num_unit=G.output_num())
F2 = ArgClassifier(num_classes=n_class, num_unit=G.output_num())

F1.apply(weights_init)
F2.apply(weights_init)

G = nn.DataParallel(G).cuda()
F1 = nn.DataParallel(F1).cuda()
F2 = nn.DataParallel(F2).cuda()
"""
optimizer_g = torch.optim.SGD(list(G.module.features.parameters()), lr=0.0002, weight_decay=0.00005,momentum=5e-4)
optimizer_f1 = torch.optim.SGD(list(F1.parameters()) , lr=0.002, weight_decay=0.00005, momentum=5e-4)
optimizer_h = torch.optim.SGD(list(G.module.g.parameters()) , lr=0.002, weight_decay=0.00005, momentum=5e-4)
"""
optimizer_g = torch.optim.SGD(list(G.module.features.parameters()), lr=0.0003,weight_decay=0.0005)
optimizer_f1 = torch.optim.SGD(list(F1.parameters()), lr=0.0003, weight_decay=0.0005, momentum=0.9)
optimizer_f2 = torch.optim.SGD(list(F2.parameters()), lr=0.0003, weight_decay=0.0005, momentum=0.9)
optimizer_h = torch.optim.SGD(list(G.module.g.parameters()) , lr=0.0003, weight_decay=0.0005, momentum=0.9)

train_transform = transforms.Compose(
    [transforms.RandomResizedCrop(224),  # 随机裁剪一个area然后再resize
     transforms.RandomHorizontalFlip(),  # 随机水平翻转
     transforms.ToTensor(),  # 归一化到[0.0,1.0]
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
     ])

contra_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.RandomApply([transforms.Pad(padding=(8, 16, 32, 64), fill=(255, 0, 0), padding_mode='symmetric')],p=0.5),   
    #transforms.Resize((224,224)),
    
    transforms.RandomGrayscale(p=0.2),
    #GaussianBlur(kernel_size=int(0.1 * 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),

])

test_transform = transforms.Compose(
    [transforms.Resize((288,288)),
     transforms.ToTensor(),  # 归一化到[-1.0,1.0]
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
     ])

train_data_source = GetLoader(dataroot="/home/wsco/wb_exper/image_CLEF/c", transforms=train_transform,contra_transforms=contra_transform)
data_loader_source = torch.utils.data.DataLoader(train_data_source, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=8)
train_data_target = GetLoader(dataroot="/home/wsco/wb_exper/image_CLEF/p", transforms=train_transform,contra_transforms=contra_transform)
data_loader_target = torch.utils.data.DataLoader(train_data_target, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=8)

test_data_target = GetLoader(dataroot="/home/wsco/wb_exper/image_CLEF/p", transforms=test_transform,contra_transforms=contra_transform)
data_loader_t_test = torch.utils.data.DataLoader(test_data_target, batch_size=batch_size, shuffle=False,drop_last=False, num_workers=8)

def discrepancy(t_logits1, t_logits2):
    return torch.mean(torch.abs(F.softmax(t_logits1,dim = 1) - F.softmax(t_logits2,dim = 1)))


def train_one_epoch(G, F1, F2, dataloaders_source, dataloaders_target, temperature, batch_size, epoch, n_epoch, k):
    best_acc = 0
    # for epoch in range(0,n_epoch):
    G.train()
    F1.train()

    len_data_loader = max(len(dataloaders_source), len(dataloaders_target))
    data_source_iter = iter(dataloaders_source)
    data_target_iter = iter(dataloaders_target)

    total_loss, correct = 0, 0
    for i in range(len_data_loader):
        ns1, ns2, nt1, nt2 = torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda()

        optimizer_g.zero_grad()
        optimizer_f1.zero_grad()
        optimizer_f2.zero_grad()
        optimizer_h.zero_grad()
        
        if i < len(dataloaders_source):
            data_source = data_source_iter.next()
            s_img, s_img1, s_img2, s_label = (data_source)
            s_img = s_img.cuda()
            s_img1 = s_img1.cuda()
            s_img2 = s_img2.cuda()
            s_label = s_label.cuda()

            with torch.set_grad_enabled(True):
                s_feat, s_out = G(s_img, None,F1)
                s_logits = F2(s_feat)
                #s_feat1, s_out1 = G(s_img1, None,F1)
                #s_logits1 = F2(s_feat1)
                #s_feat2, s_out2 = G(s_img2, None,F1)
                #s_logits2 = F2(s_feat2)

                loss_cross = criterion_cross(s_logits, s_label)
                #dis_loss_s = discrepancy(s_logits1,s_logits2)
                #loss_contra_source = criterion_contra(s_out1, s_out2, temperature, batch_size)

                """
                if epoch > 0:
                    ns1,_ = neg(s_logits)
                    #ns2,_ = neg(s_logits2)
                """
            
        if i < len(dataloaders_target):
            data_target = data_target_iter.next()
            t_img, t_img1, t_img2, t_label = (data_target)
            t_img = t_img.cuda()
            t_img1 = t_img1.cuda()
            t_img2 = t_img2.cuda()
            t_label = t_label.cuda()
            #t_img,t_img1,t_img2,t_label = Variable(t_img),Variable(t_img1),Variable(t_img2),Variable(t_label)
            with torch.set_grad_enabled(True):
                t_feat, t_out = G(t_img, None,F1)
                t_logits = F1(t_feat)
                t_feat1, t_out1 = G(t_img1, None,F1)
                t_logits1 = F2(t_feat1)
                t_feat2, t_out2 = G(t_img2, None,F1)
                t_logits2 = F2(t_feat2)

                #loss_contra_target = criterion_contra(t_out1, t_out2, temperature, batch_size)
                dis_loss_t = discrepancy(t_logits1,t_logits2)
                #loss_gra = gradient_discrepancy_loss_margin(loss_contra_source, loss_contra_target, G)
                
                if epoch > 0:
                    nt1, _ = neg(t_logits)
                    #nt2, _ = neg(t_logits2)
                
        loss = loss_cross - dis_loss_t + nt1#+ nt2
        loss.backward()
        optimizer_f2.step()
        
        
        optimizer_g.zero_grad()
        optimizer_f1.zero_grad()
        optimizer_f2.zero_grad()
        optimizer_h.zero_grad()
        
        if i < len(dataloaders_source):
            """
            data_source = data_source_iter.next()           
            s_img, s_img1, s_img2, s_label = tqdm(data_source)
            s_img = s_img.cuda()
            s_img1 = s_img1.cuda()
            s_img2 = s_img2.cuda()
            s_label = s_label.cuda()
            """
            with torch.set_grad_enabled(True):
                s_feat, s_out = G(s_img, s_label,F1)
                s_logits = F1(s_feat)
                s_feat1, s_out1 = G(s_img1, s_label,F1)
                #s_logits11 = F2(s_feat1)
                #s_logits1 = F1(s_feat1)
                s_feat2, s_out2 = G(s_img2, s_label,F1)
                #s_logits2 = F1(s_feat2)
                #s_logits22 = F2(s_feat2)

                loss_cross = criterion_cross(s_logits, s_label)

                loss_contra_source = criterion_contra(s_out1, s_out2, temperature, batch_size)
                #dis_loss_s = discrepancy(s_logits11,s_logits22)
                """
                if epoch > 0:
                    ns1,_ = neg(s_logits)
                    #ns2,_ = neg(s_logits2)
                """
            preds = torch.max(s_logits, 1)[1]
            correct += torch.sum(preds == s_label.data)
        if i < len(dataloaders_target):
            """
            data_target = data_target_iter.next()  
            t_img, t_img1, t_img2, t_label = tqdm(data_target)
            t_img = t_img.cuda()
            t_img1 = t_img1.cuda()
            t_img2 = t_img2.cuda()
            t_label = t_label.cuda()
            #t_img,t_img1,t_img2,t_label = Variable(t_img),Variable(t_img1),Variable(t_img2),Variable(t_label)
            """
            with torch.set_grad_enabled(True):
                t_feat, t_out = G(t_img, None,F1)
                t_logits = F1(t_feat)
                t_feat1, t_out1 = G(t_img1, None,F1)
                t_logits1 = F1(t_feat1)
                t_logits11 = F2(t_feat1)
                t_feat2, t_out2 = G(t_img2, None,F1)
                #t_logits2 = F1(t_feat2)
                t_logits22 = F2(t_feat2)

                loss_contra_target = criterion_contra(t_out1, t_out2, temperature, batch_size)
                dis_loss_t = discrepancy(t_logits11,t_logits22)
                #loss_gra = gradient_discrepancy_loss_margin(loss_contra_source, loss_contra_target, G)
                
                if epoch > 0:
                    nt1, _ = neg(t_logits)
                    #nt2, _ = neg(t_logits2)
                

        loss =  2.5 * loss_cross + 2.0 * (loss_contra_source + loss_contra_target) + dis_loss_t + nt1#+ loss_gra
        loss.backward()

        optimizer_g.step()
        optimizer_f1.step()
        optimizer_h.step()       
        
        total_loss += loss.item() * (s_img.size(0) + t_img.size(0)) / (4.5 + 2.5 + 2.0)
        loss_cross ,loss_contra_source,loss_contra_target,dis_loss_s,dis_loss_t = torch.tensor(0.).cuda(),torch.tensor(0.).cuda(),torch.tensor(0.).cuda(),torch.tensor(0.).cuda(),torch.tensor(0.).cuda()
    epoch_loss = total_loss / (len(dataloaders_source.dataset) + len(dataloaders_target.dataset))
    ls_loss.append(epoch_loss)
    epoch_acc = correct.double() / len(dataloaders_source.dataset)
    #epoch_acc2 = correct2.double() / len(dataloaders_source.dataset)
    print(f'Epoch: [{epoch:02d}/{n_epoch:02d}]----, loss: {epoch_loss:.6f},【{correct}|{len(dataloaders_source.dataset)}】,acc: {epoch_acc:.4f}')

def test(epoch, G, F1, dataloader):
    G.eval()
    F1.eval()
    correct = 0
    #len_target_dataset = len(dataloader.dataset)
    with torch.no_grad():
        for data,data1,data2,target in tqdm(dataloader):
            data,data1,data2, target = data.cuda(),data1.cuda(),data2.cuda(), target.cuda()
            feat,_ = G(data,None,F1)
            output = F1(feat)
            pred = torch.max(output, 1)[1]
            correct += torch.sum(pred == target.data)
            #print('pred:',pred)
            #print('label:',target)
    acc = correct.double() / len(dataloader.dataset)
    ls_acc.append(acc.item()*100)
    print(f'Epoch:{epoch:02d},【{correct}|{len(dataloader.dataset)}】,acc:{acc:.4f}')
    return acc

#冻结某些层
def set_freeze_by_names(G, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in G.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze

def freeze_by_names(G, layer_names):
    set_freeze_by_names(G, layer_names, True)

def unfreeze_by_names(G, layer_names):
    set_freeze_by_names(G, layer_names, False)

#冻结'conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3'等层
freeze_by_names(G, ('conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2'))

temperature = 0.5
best_acc = 0
k = 1.9
ls_loss = []
ls_acc = []
for epoch in range(n_epoch):

    train_one_epoch(G, F1, F2, data_loader_source, data_loader_target, temperature, batch_size, epoch, n_epoch,k)
    test_acc = test(epoch, G, F1, data_loader_t_test)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(G, 'result/model_G.pth')
    print(f'Best acc is :{best_acc:.4f}')

"""
with open("/home/wsco/wb_exper/train_on_img_clef/ls_loss.txt","w") as fp:
    [fp.write(str(item)+'\n') for  item in ls_loss]
    fp.close()
with open("/home/wsco/wb_exper/train_on_img_clef/ls_acc.txt","w") as fp:
    [fp.write(str(item)+'\n') for  item in ls_acc]
    fp.close()
"""