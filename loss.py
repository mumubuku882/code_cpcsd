import torch
import torch.nn.functional as F
#import ipdb
from torch.autograd import grad
from torch.autograd import Variable
from torch import nn
    
class Cross_entropy_loss(torch.nn.Module):
    def __init__(self):
        super(Cross_entropy_loss,self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(self,pred,labels):

        return self.criterion(pred,labels)

class InfoNCE_loss(torch.nn.Module):
    def __init__(self):
        super(InfoNCE_loss, self).__init__()

    def forward(self,out_1,out_2,temperature,batch_size):
        
        out = torch.cat([out_1, out_2], dim=0)#64 * 128
        //删除部分代码,如需使用请与作者联系

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)#32
        # [2*B]
        
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)#64
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()#sim_matrix.sum(dim=-1)为 64维度
        #ipdb.set_trace()
        return loss

   
"""     
class InfoNCE_loss(torch.nn.Module):
    def __init__(self):
        super(InfoNCE_loss, self).__init__()
        self.MI_threshold = 0.53

    def forward(self,z1,z2,temperature,batch_size):
        
        z = torch.cat([z1, z2], dim=0)
        # print(z.shape)
        n_samples = z.shape[0]

        cov = torch.mm(z, z.t().contiguous())
        # print('loss')
        # print(cov)
        sim = torch.exp(cov / temperature)
        # block = torch.eye(n_samples//2)
        # half = torch.cat([block, block], dim=0)
        # full = torch.cat([half, half], dim=1)
        full = torch.eye(n_samples)
        mask = torch.eq(full, 0).to(sim.device)
        # print(mask)
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)
        # print(neg)
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature) 
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / neg).mean()
        return loss
"""

class Neg_MI(torch.nn.Module):
    def __init__(self):
        super(Neg_MI, self).__init__()
        self.MI_threshold = 0.53

    def entropy(self,logits):
        p = F.softmax(logits, dim=-1)
        return -torch.sum(p * torch.log(p), dim=-1).mean()

    def neg_mutual_information(self,logits):
        condi_entropy = self.entropy(logits)
        y_dis = torch.mean(F.softmax(logits, dim=-1), dim=0)
        y_entropy = (-y_dis * torch.log(y_dis)).sum()
        if y_entropy.item() < self.MI_threshold:
            return -y_entropy + condi_entropy, y_entropy
        else:
            return condi_entropy, y_entropy
            
    def forward(self,logits):
        return self.neg_mutual_information(logits)

"""
#MMD loss
class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss
"""
def gradient_discrepancy_loss_margin(loss_contra_s, loss_contra_t, net_g):
    
    gm_loss = 0
    grad_cossim11 = []
  
    for n, p in net_g.named_parameters():
       
        real_grad = grad([loss_contra_s],
                            [p],
                            create_graph=True,
                            only_inputs=True,
                            allow_unused=False)[0]
        fake_grad = grad([loss_contra_t],
                            [p],
                            create_graph=True,
                            only_inputs=True,
                            allow_unused=False)[0]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
        #_mse = F.mse_loss(fake_grad, real_grad)
        grad_cossim11.append(_cossim)
        #grad_mse.append(_mse)

    grad_cossim1 = torch.stack(grad_cossim11)
    gm_loss = (1.0 - grad_cossim1).mean()
        
    return gm_loss

class contrastive_loss(nn.Module):
    def __init__(self, tau=0.05, normalize=False):
        super(contrastive_loss, self).__init__()
        self.tau = tau
        self.normalize = normalize

    def forward(self, xi, xj):

        x = torch.cat((xi, xj), dim=0)

        is_cuda = x.is_cuda
        sim_mat = torch.mm(x, x.T)
        if self.normalize:
            sim_mat_denom = torch.mm(torch.norm(x, dim=1).unsqueeze(1), torch.norm(x, dim=1).unsqueeze(1).T)
            sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)

        sim_mat = torch.exp(sim_mat / self.tau)

        # no diag because it's not diffrentiable -> sum - exp(1 / tau)
        # diag_ind = torch.eye(xi.size(0) * 2).bool()
        # diag_ind = diag_ind.cuda() if use_cuda else diag_ind

        # sim_mat = sim_mat.masked_fill_(diag_ind, 0)

        # top
        if self.normalize:
            sim_mat_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / sim_mat_denom / self.tau)
        else:
            sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.tau)

        sim_match = torch.cat((sim_match, sim_match), dim=0)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.tau)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))

        return loss
