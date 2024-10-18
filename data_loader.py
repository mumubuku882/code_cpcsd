import torch.utils.data as data
from PIL import Image
import os


class GetLoader(data.DataLoader):
    def __init__(self, dataroot, transforms , contra_transforms):
        self.dataroot = dataroot
        self.transform = transforms
        self.contra_transform = contra_transforms
        self.img_path = []
        self.img_label = []
        l = 0
        self.n_data = 0
        for filename in sorted(os.listdir(self.dataroot),key=lambda x: x[0:6]):
            file_path = os.path.join(self.dataroot, filename)
            for imgname in os.listdir(file_path):
                img_path = os.path.join(self.dataroot, filename, imgname)
                self.img_path.append(img_path)
                self.img_label.append(l)
                self.n_data += 1
                #print(img_path,l)
            l = l + 1

    def __getitem__(self, item):
        img_paths, labels = self.img_path[item], self.img_label[item]
        imgs = Image.open(img_paths).convert('RGB')
        if self.transform is not None:
            argue_img = self.transform(imgs)
            pos_1 = self.contra_transform(imgs)
            pos_2 = self.contra_transform(imgs)
            labels = int(labels)

        return argue_img, pos_1, pos_2, labels

    def __len__(self):

        return self.n_data
