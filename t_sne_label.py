from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as cm
import torch

"""
dl_source_train_1000 = torch.utils.data.DataLoader(ds_source, 1000)
dl_target_train_1000 = torch.utils.data.DataLoader(ds_target, 1000)
source_train_1000 , _ = next(iter(dl_source_train_1000))
target_train_1000 , _ =next(iter(dl_target_train_1000))
"""
def class_label(s_feat,s_label,t_feat,t_label,num_class,num_sum):

    flgs = [0] * num_class
    flgt = [0] * num_class
    featS = [torch.zeros([1,2])] * num_class
    featT = [torch.zeros([1,2])] * num_class

    for i in range(num_sum):
        for j in range(num_class):
            if s_label[i] == j:
                if flgs[j] == 0:
                    featS[j] = torch.unsqueeze(torch.tensor(s_feat[i,:]),0)
                    flgs[j] = 1
                else:
                    featS[j] = torch.cat((featS[j],torch.unsqueeze(torch.tensor(s_feat[i,:]),0)), dim=0)
            if t_label[i] == j:
                if flgt[j] == 0:
                    featT[j] = torch.unsqueeze(torch.tensor(t_feat[i,:]),0)
                    flgt[j] = 1
                else:
                    featT[j] = torch.cat((featT[j],torch.unsqueeze(torch.tensor(t_feat[i,:]),0)), dim=0)
    return featS,featT



def plot_distribution(source_train_1000,target_train_1000,s_label,t_label,num_class,num_sum):
    # Create a two dimensional t-SNE projection of the embeddings
    tsne = TSNE(perplexity = 30, n_components = 2,init='pca',learning_rate = 700, n_iter=5000,verbose=1)
   
    if source_train_1000.is_cuda:
        source_train_1000 = source_train_1000.cpu().detach().numpy()
    else:
        source_train_1000 = source_train_1000.detach().numpy()
    if target_train_1000.is_cuda:
        target_train_1000 = target_train_1000.cpu().detach().numpy()
    else:
        target_train_1000 = target_train_1000.detach().numpy()
    
    tsne_source = tsne.fit_transform(source_train_1000.reshape(600,-1))#转二维数组
    tsne_target = tsne.fit_transform(target_train_1000.reshape(600,-1))
    
    featS,featT = class_label(tsne_source,s_label,tsne_target,t_label,num_class,num_sum)
    
    # Plot those points as a scatter plot and label them based on the pred labels
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 12
    colors = ['red','green','pink','blue','yellow','black']
    for lab in range(num_categories):       
        
        colors = ['red','green','pink','blue','yellow','black','orange','brown','gray','purple','navy','teal']
        ax.scatter(featS[lab][:,0].detach().numpy(),featS[lab][:,1].detach().numpy(), c=colors[lab], label = '' ,alpha=0.5)
        ax.scatter(featT[lab][:,0].detach().numpy(),featT[lab][:,1].detach().numpy(), c=colors[lab], label = '' ,alpha=0.5)   
    ax.legend(fontsize='large', markerscale=2)
    plt.show()
    
def draw_t_sne(G,F1, dataloaders_source, dataloaders_target):
    G.eval()
    len_data_loader = max(len(dataloaders_source), len(dataloaders_target))
    data_source_iter = iter(dataloaders_source)
    data_target_iter = iter(dataloaders_target)
    
    for i in range(len_data_loader):

        if i < len(dataloaders_source):
            data_source = data_source_iter.next()
            s_img, _, _, s_label = (data_source)
            s_img = s_img.cuda()
            s_label = s_label.cuda()

            with torch.set_grad_enabled(False):
                s_feat_batch, s_out = G(s_img, None, F1)
            if i == 0:
                s_feat = s_feat_batch
                s_label_total = s_label
            else:
                s_feat = torch.cat((s_feat,s_feat_batch), dim=0)
                s_label_total = torch.cat((s_label_total,s_label), dim = 0)
        if i < len(dataloaders_target):
            data_target = data_target_iter.next()
            t_img, _, _, t_label = (data_target)
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            with torch.set_grad_enabled(False):
                t_feat_batch, t_out = G(t_img, None, F1)
            if i == 0:
                t_feat = t_feat_batch
                t_label_total = t_label
            else:
                t_feat = torch.cat((t_feat,t_feat_batch), dim=0)
                t_label_total = torch.cat((t_label_total,t_label), dim = 0)
                
    plot_distribution(s_feat, t_feat,s_label_total,t_label_total,num_class=12,num_sum=600)

