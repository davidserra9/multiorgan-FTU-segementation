from torch.utils.data import Dataset
import os
import cv2
from tools.visualize import read_img_and_mask
from tqdm import tqdm

class FTUs(Dataset):
    """
    This class is used to load the data from the FTUs dataset.
    """
    def __init__(self, root_path, split, transform=None):
        self.root_path = root_path
        self.split = split
        self.transform = transform

        # read imgs and masks from split
        path_imgs = os.path.join(self.root_path, 'train_images')
        self.imgs = []
        self.masks = []

        # todo: hacerlo fuera y cargar pickles que sino peta la ram y asi no hay q hacerlo cada vez q se crea dataset
        print(f'Loading images to dataset...')
        if split == 'train':
            for img in tqdm(os.listdir(path_imgs)):
                # read img and mask
                # get the name of the image without the extension
                name = img.split('.')[0]
                img, mask = read_img_and_mask(name, path_db=self.root_path, show=False)
                self.imgs.append(img)
                self.masks.append(mask)
        else:
            for img in os.listdir(path_imgs):
                # read img and mask
                img = cv2.imread(img)
                self.imgs.append(img)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # idx must be lower than __len__
        if idx >= len(self.imgs):
            raise IndexError('Index out of range')

        # load the image and the mask
        img = self.imgs[idx]
        mask = None

        if self.split == 'train':
            mask = self.masks[idx]

        if self.transform:
            img = self.transform(img)

        return img, mask


if __name__ == "__main__":
    db = FTUs(root_path='../data/hubmap-organ-segmentation', split='train')
    print('finished')