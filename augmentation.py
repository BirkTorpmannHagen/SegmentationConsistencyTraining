import albumentations as alb
import torch
import torch.nn as nn


class Augmentations(nn.Module):
    def __init__(self):
        super(Augmentations, self).__init__()
        self.pixelwise_augments = alb.Compose([
            alb.ColorJitter(brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.05, p=1),
            alb.GaussNoise(var_limit=0.01, p=1),
            alb.ImageCompression(quality_lower=10,
                                 quality_upper=100,
                                 p=1)
        ]
        )

        self.geometric_augments = alb.Compose([alb.RandomRotate90(p=1),
                                 alb.Flip(p=1),
                                 alb.OpticalDistortion(distort_limit=1, p=1)])

    def forward(self, image, mask):
        augmented_imgs = torch.zeros_like(image)
        augmented_masks = torch.zeros_like(mask)
        for batch_idx in range(image.shape[0]):
            aug_img = image[batch_idx].squeeze().cpu().numpy().T
            aug_mask = mask[batch_idx].squeeze().cpu().numpy().T
            pixelwise = self.pixelwise_augments(image=aug_img)["image"]
            geoms = self.geometric_augments(image=pixelwise, mask=aug_mask)
            augmented_imgs[batch_idx] = torch.Tensor(geoms["image"].T)
            augmented_masks[batch_idx] = torch.Tensor(geoms["mask"].T)
        return augmented_imgs, augmented_masks
