import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('/home/wsco/wb_exper/train_on_img_clef/ls_loss.txt') #忽略第一行，分隔符
y = data
x = np.arange(0,200,1)

fig = plt.figure(figsize=[8,6]) #默认 6.4 x 4.8
sub = fig.add_subplot(111)


sub.plot(x,y)

sub.set_xlabel('Number of Epoches',fontsize = 14)
sub.set_ylabel('Classification Loss',fontsize = 14)
sub.set_xticks([0,25,50,75,100,125,150,175,200]) #主刻度
sub.set_yticks([0,0.5,1.0,1.5,2.0,2.5]) #主刻度

sub.grid(axis = 'both',which = 'both',linestyle = '--' )#网格

sub.tick_params(axis = 'both',which = 'major',direction = 'out', length = 5, width = 2,grid_alpha = 0.5)

plt.show()
fig.savefig('/home/wsco/wb_exper/train_on_img_clef/loss.jpg',dpi=1000) # dpi分辨率，默认100
plt.close('all')