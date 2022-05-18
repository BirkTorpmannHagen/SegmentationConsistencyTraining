from os import listdir
from os.path import join
import numpy as np
import PIL.Image
from PIL.Image import open
from torch.utils.data import Dataset
from torchvision import transforms
from augmentation import *

def pipeline_tranforms():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

class KvasirSegmentationDataset(Dataset):
    """
        Dataset class that fetches images with the associated segmentation mask.
    """

    def __init__(self, path, split="train"):
        super(KvasirSegmentationDataset, self).__init__()
        self.path = join(path, "segmented-images/")
        self.fnames = listdir(join(self.path, "images"))
        self.common_transforms = pipeline_tranforms()
        self.split = split
        train_size = int(len(self.fnames) * 0.8)
        val_size = (len(self.fnames) - train_size) // 2
        test_size = len(self.fnames) - train_size - val_size
        self.fnames_train = self.fnames[:train_size]
        self.fnames_val = self.fnames[train_size:train_size + val_size]
        self.fnames_test = self.fnames[train_size + val_size:]
        self.split_fnames = None  # iterable for selected split
        if self.split == "train":
            self.size = train_size
            self.split_fnames = self.fnames_train
        elif self.split == "val":
            self.size = val_size
            self.split_fnames = self.fnames_val
        elif self.split == "test":
            self.size = test_size
            self.split_fnames = self.fnames_test
        else:
            raise ValueError("Choices are train/val/test")

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        image = np.array(open(join(self.path, "images/", self.split_fnames[index])).convert("RGB"))
        mask = np.array(open(join(self.path, "masks/", self.split_fnames[index])).convert("L"))
        image = self.common_transforms(PIL.Image.fromarray(image))
        mask = self.common_transforms(PIL.Image.fromarray(mask))
        mask = (mask > 0.5).float()
        return image, mask, self.split_fnames[index]


class KvasirWithAugments(KvasirSegmentationDataset):
    def __init__(self, path, split):
        super(KvasirWithAugments, self).__init__(path, split)
        self.augments = Augmentations()
        self.p = 0.5

    def __getitem__(self, index):
        image = np.array(open(join(self.path, "images/", self.split_fnames[index])).convert("RGB"))
        mask = np.array(open(join(self.path, "masks/", self.split_fnames[index])).convert("L"))
        image = self.common_transforms(PIL.Image.fromarray(image))
        mask = self.common_transforms(PIL.Image.fromarray(mask))
        mask = (mask > 0.5).float()
        if self.split == "train" and np.random.rand() < self.p:
            image, mask = self.augments(image.unsqueeze(0), mask.unsqueeze(0))
            image = image.squeeze()
            mask = mask.squeeze(0)
        return image, mask, self.split_fnames[index]

    def set_prob(self, prob):
        self.p = prob


class EtisDataset(Dataset):
    """
        Dataset class that fetches Etis-LaribPolypDB images with the associated segmentation mask.
        Used for testing.
    """
    def __init__(self, path):
        super(EtisDataset, self).__init__()
        self.path = path
        self.len = len(listdir(join(self.path, "ETIS-LaribPolypDB")))
        self.common_transforms = pipeline_tranforms()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image = self.common_transforms(
            open(join(self.path, "ETIS-LaribPolypDB/{}.jpg".format(index + 1))).convert("RGB"))
        mask = self.common_transforms(
            open(join(self.path, "GroundTruth/p{}.jpg".format(index + 1))).convert("RGB"))
        mask = (mask > 0.5).float()
        return image, mask, index + 1


class CVC_ClinicDB(Dataset):
    def __init__(self, root_directory):
        super(CVC_ClinicDB, self).__init__()
        self.root = root_directory
        self.mask_fnames = listdir(join(self.root, "Ground Truth"))
        self.mask_locs = [join(self.root, "Ground Truth", i) for i in self.mask_fnames]
        self.img_locs = [join(self.root, "Original", i) for i in
                         self.mask_fnames]
        self.common_transforms = pipeline_tranforms()

    def __getitem__(self, idx):
        mask = self.common_transforms(open(self.mask_locs[idx]))
        image = self.common_transforms(open(self.img_locs[idx]))
        return image, mask, self.mask_fnames[idx]

    def __len__(self):
        return len(self.mask_fnames)


class EndoCV2020(Dataset):
    def __init__(self, root_directory):
        super(EndoCV2020, self).__init__()
        self.root = root_directory
        self.mask_fnames = listdir(join(self.root, "masksPerClass", "polyp"))
        self.mask_locs = [join(self.root, "masksPerClass", "polyp", i) for i in self.mask_fnames]
        self.img_locs = [join(self.root, "originalImages", i.replace("_polyp", "").replace(".tif", ".jpg")) for i in
                         self.mask_fnames]
        self.common_transforms = pipeline_tranforms()

    def __getitem__(self, idx):
        mask = self.common_transforms(open(self.mask_locs[idx]))
        image = self.common_transforms(open(self.img_locs[idx]))
        return image, mask, self.mask_fnames[idx]

    def __len__(self):
        return len(self.mask_fnames)

