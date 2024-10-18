from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import torch.nn.init as init
from torch.autograd import Function
from backbone import *


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lambd


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert (padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResBottle(nn.Module):
    def __init__(self, option='resnet50', pret=True, feature_dim = 128):
        super(ResBottle, self).__init__()
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        self.model_ft = model_ft
        mod = list(model_ft.children())
        mod.pop()
        # self.model_ft =model_ft
        self.features = nn.Sequential(*mod)

        self.dim = 2048
        num_fc_features = model_ft.fc.in_features
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def get_toalign_weight(self, f, label=None):
        assert label is not None, f'label should be asigned'
        w = self.Ff.module.fc.weight[label].detach()  # [B, C]

        #print(self.Ff.module.fc.weight.shape)
        """
        if self.hda:
            w0 = self.fc0.weight[labels].detach()
            w1 = self.fc1.weight[labels].detach()
            w2 = self.fc2.weight[labels].detach()
            w = w - (w0 + w1 + w2)
        """
        eng_org = (f ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        eng_aft = ((f * w) ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        scalar = (eng_org / eng_aft).sqrt()
        w_pos = w * scalar

        return w_pos

    def forward(self, x, label, Ff):
        self.Ff = Ff
        x = self.features(x) #全连接层之前的特征 2048
        #print(x.shape) (16,2048,1,1)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        """
        x = x.view(x.size(0), -1)

        x = self.bottleneck(x) #全连接层转为 256维

        x = x.view(x.size(0), self.dim)
        """

        if label is not None:
            object_feature = x * self.get_toalign_weight(x, label)
        else:
            object_feature = x
        
        out = self.g(object_feature)

        return x, F.normalize(out, dim=-1)

    def output_num(self):
        return self.dim


class ResNet_all(nn.Module):
    def __init__(self, option='resnet18', pret=True):
        super(ResNet_all, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        # mod = list(model_ft.children())
        # mod.pop()
        # self.model_ft =model_ft
        self.conv1 = model_ft.conv1
        self.bn0 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool
        self.layer1 = model_ft.layer1
        self.layer2 = model_ft.layer2
        self.layer3 = model_ft.layer3
        self.layer4 = model_ft.layer4
        self.pool = model_ft.avgpool
        self.fc = nn.Linear(2048, 12)

    def forward(self, x, layer_return=False, input_mask=False, mask=None, mask2=None):
        if input_mask:
            x = self.conv1(x)
            x = self.bn0(x)
            x = self.relu(x)
            conv_x = x
            x = self.maxpool(x)
            fm1 = mask * self.layer1(x)
            fm2 = mask2 * self.layer2(fm1)
            fm3 = self.layer3(fm2)
            fm4 = self.pool(self.layer4(fm3))
            x = fm4.view(fm4.size(0), self.dim)
            x = self.fc(x)
            return x  # ,fm1
        else:
            x = self.conv1(x)
            x = self.bn0(x)
            x = self.relu(x)
            conv_x = x
            x = self.maxpool(x)
            fm1 = self.layer1(x)
            fm2 = self.layer2(fm1)
            fm3 = self.layer3(fm2)
            fm4 = self.pool(self.layer4(fm3))
            x = fm4.view(fm4.size(0), self.dim)
            x = self.fc(x)
            if layer_return:
                return x, fm1, fm2
            else:
                return x


class ResClassifier(nn.Module):
    def __init__(self, num_classes=12, num_unit=2048):
        super(ResClassifier, self).__init__()
        #layers = []
        self.fc = nn.Linear(num_unit,num_classes,bias = True)
        self.fc.weight.data.normal_(0, 0.005)
        self.fc.bias.data.fill_(0.1)
        # currently 10000 units
        

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.fc(x)
        return x

class Predictor(nn.Module):
    def __init__(self, num_classes=12, num_unit=2048, middle = 1024):
        super(Predictor, self).__init__()
        #layers = []
        self.fc1 = nn.Linear(num_unit,middle,bias = True)
        self.fc1.weight.data.normal_(0, 0.005)
        self.fc1.bias.data.fill_(0.1)
        self.fc2 = nn.Linear(middle,num_classes,bias = True)
        self.fc2.weight.data.normal_(0, 0.005)
        self.fc2.bias.data.fill_(0.1)
        # currently 10000 units
        

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class ArgClassifier(nn.Module):
    def __init__(self, num_classes=12, num_layer=1, num_unit=2048, prob=0.5, middle=1000):
        super(ArgClassifier, self).__init__()
        layers = []
        # currently 10000 units
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer - 1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle, middle))
            layers.append(nn.BatchNorm1d(middle, affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(middle, num_classes))
        self.classifier = nn.Sequential(*layers)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x
        
        
        
class ReSSL(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, backbone='resnet50', dim=512, K=65536*2, m=0.999):
        """
        dim: feature dimension (default: 512)
        K: queue size; number of negative keys (default: 65536*2)
        m: moco momentum of updating key encoder (default: 0.999)
        """
        super(ReSSL, self).__init__()

        self.K = K
        self.m = m

        # create the encoders
        self.encoder_q = BackBone(backbone=backbone, dim=dim)
        self.encoder_k = BackBone(backbone=backbone, dim=dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: contrastive augmented image
            im_k: weak augmented image
        Output:
            logitsq, logitsk
        """

        q = self.encoder_q(im_q)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.encoder_k(im)  # keys: NxC
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        logitsq = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logitsk = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return logitsq, logitsk


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output

