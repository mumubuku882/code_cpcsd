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
def plot_distribution(source_train_1000,target_train_1000):
    # Create a two dimensional t-SNE projection of the embeddings
    tsne = TSNE(perplexity = 30, n_components = 2,learning_rate = 300, n_iter=5000,verbose=1)
    if source_train_1000.is_cuda:
        source_train_1000 = source_train_1000.cpu().detach().numpy()
    else:
        source_train_1000 = source_train_1000.detach().numpy()
    if target_train_1000.is_cuda:
        target_train_1000 = target_train_1000.cpu().detach().numpy()
    else:
        target_train_1000 = target_train_1000.detach().numpy()
    tsne_source = tsne.fit_transform(source_train_1000.reshape(600,-1))
    tsne_target = tsne.fit_transform(target_train_1000.reshape(600,-1))
    #print("shape:", tsne_source.shape)
    # Plot those points as a scatter plot and label them based on the pred labels
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 2
    for lab in range(num_categories):
        if lab ==0:
            ax.scatter(tsne_source[:,0],tsne_source[:,1], c='red', label = 'source' ,alpha=0.5)
        else:
            ax.scatter(tsne_target[:,0],tsne_target[:,1], c='blue', label = 'target' ,alpha=0.5)
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
            else:
                s_feat = torch.cat((s_feat,s_feat_batch), dim=0)
        if i < len(dataloaders_target):
            data_target = data_target_iter.next()
            t_img, _, _, t_label = (data_target)         
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            with torch.set_grad_enabled(False):
                t_feat_batch, t_out = G(t_img, None, F1)
            if i == 0:
                t_feat = t_feat_batch
            else:
                t_feat = torch.cat((t_feat,t_feat_batch), dim=0)
                
    plot_distribution(s_feat, t_feat)
