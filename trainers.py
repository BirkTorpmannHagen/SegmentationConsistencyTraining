import numpy as np
import torch.optim.optimizer
from torch.utils.data import DataLoader
from dataloaders import KvasirSegmentationDataset, KvasirWithAugments, EtisDataset
from metrics import iou
from consistency_losses import *
from augmentation import Augmentations
import segmentation_models

class VanillaTrainer:
    def __init__(self, id, config):
        """

        :param model: String describing the model type. Can be DeepLab, TriUnet, ... TODO
        :param config: Contains hyperparameters : lr, epochs, batch_size, T_0, T_mult
        """
        self.config = config
        self.device = config["device"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.model = None
        self.id = id
        self.model_str = config["model"]
        self.mnv = Augmentations()
        self.predictor_name = f"Predictors/No_Augmentation_{self.model_str}-{self.id}"

        if self.model_str == "DeepLab":
            self.model = segmentation_models.DeepLab().to(self.device)
        elif self.model_str == "TriUnet":
            self.model = segmentation_models.TriUnet().to(self.device)
        elif self.model_str == "Unet":
            self.model = segmentation_models.Unet().to(self.device)
        elif self.model_str == "FPN":
            self.model = segmentation_models.FPN().to(self.device)


        else:
            raise AttributeError("model_str not valid; choices are DeepLab, TriUnet, InductiveNet, FPN, Unet")

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = vanilla_losses.JaccardLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2)
        self.train_set = KvasirSegmentationDataset("Datasets/HyperKvasir", split="train")
        self.val_set = KvasirSegmentationDataset("Datasets/HyperKvasir", split="val")
        self.test_set = KvasirSegmentationDataset("Datasets/HyperKvasir", split="test")

    def train_epoch(self):
        self.model.train()
        losses = []
        for x, y, fname in DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True):
            image = x.to(self.device)
            mask = y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, mask)
            loss.backward()
            self.optimizer.step()
            losses.append(np.abs(loss.item()))
        return np.mean(losses)

    def train(self):
        best_val_loss = 10
        print("Starting Segmentation training")

        for i in range(self.epochs):
            training_loss = np.abs(self.train_epoch())
            val_loss, ious = self.validate(epoch=i, plot=False)
            gen_ious = self.validate_generalizability(epoch=i, plot=False)
            mean_iou = float(torch.mean(ious))
            gen_iou = float(torch.mean(gen_ious))
            test_ious = np.mean(self.test().numpy())
            self.scheduler.step(i)
            print(
                f"Epoch {i} of {self.epochs} \t"
                f" lr={[group['lr'] for group in self.optimizer.param_groups]} \t"
                f" loss={training_loss} \t"
                f" val_loss={val_loss} \t"
                f" ood_iou={gen_iou}\t"
                f" val_iou={mean_iou} \t"
                f" gen_prop={gen_iou / mean_iou}"
            )
            if val_loss < best_val_loss:
                test_ious = self.test()
                best_val_loss = val_loss
                print(f"Saving new best model. IID test-set mean iou: {float(np.mean(test_ious.numpy()))}")
                torch.save(self.model.state_dict(),self.predictor_name)

    def test(self):
        self.model.eval()
        ious = torch.empty((0,))
        with torch.no_grad():
            for x, y, fname in DataLoader(self.test_set):
                image = x.to(self.device)
                mask = y.to(self.device)
                output = self.model(image)
                batch_ious = torch.Tensor([iou(output_i, mask_j) for output_i, mask_j in zip(output, mask)])
                ious = torch.cat((ious, batch_ious.flatten()))
        return ious

    def validate(self, epoch, plot=False):
        self.model.eval()
        losses = []
        ious = torch.empty((0,)).to(self.device)
        with torch.no_grad():
            for x, y, fname in DataLoader(self.val_set):
                image = x.to(self.device)
                mask = y.to(self.device)
                aug_img, aug_mask = self.mnv(image, mask)
                output = self.model(image)
                aug_output = self.model(aug_img)

                batch_ious = torch.mean(iou(output, mask))
                loss = self.criterion(output, mask)
                losses.append(np.abs(loss.item()))
                ious = torch.cat((ious, batch_ious.flatten()))
        avg_val_loss = np.mean(losses)
        return avg_val_loss, ious

    def validate_generalizability(self, epoch, plot=False):
        self.model.eval()
        ious = torch.empty((0,)).to(self.device)
        with torch.no_grad():
            for x, y, index in DataLoader(EtisDataset("Datasets/ETIS-LaribPolypDB")):
                image = x.to(self.device)
                mask = y.to(self.device)
                output = self.model(image)
                batch_ious = torch.mean(iou(output, mask))
                ious = torch.cat((ious, batch_ious.flatten()))
            return ious


class ConsistencyTrainer(VanillaTrainer):
    def __init__(self, id, config):
        super(ConsistencyTrainer, self).__init__(id, config)
        self.consistency_criterion = ConsistencyTrainingLoss(adaptive=True).to(self.device)
        self.nakedcloss = SIL()
        self.predictor_name = f"Predictors/Consistency_Training_{self.model_str}-{self.id}"


    def train_epoch(self):
        self.model.train()
        losses = []
        for x, y, fname in DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True):
            image = x.to(self.device)
            mask = y.to(self.device)
            aug_img, aug_mask = self.mnv(image, mask)
            self.optimizer.zero_grad()
            output = self.model(image)
            aug_output = self.model(aug_img)
            mean_iou = torch.mean(iou(output, mask))
            loss = self.consistency_criterion(aug_mask, mask, aug_output, output, mean_iou)
            loss.backward()
            self.optimizer.step()
            losses.append(np.abs(loss.item()))
        return np.mean(losses)


class AugmentationTrainer(ConsistencyTrainer):
    """
        Uses vanilla data augmentation with p=0.5 instead of a a custom loss
    """

    def __init__(self, id, config):
        super(AugmentationTrainer, self).__init__(id, config)
        self.dataset = KvasirWithAugments("Datasets/HyperKvasir", "train")
        self.predictor_name = f"Predictors/Augmentation_{self.model_str}-{self.id}"







