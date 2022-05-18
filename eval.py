import numpy as np
from dataloaders import KvasirSegmentationDataset, EtisDataset
from dataloaders import CVC_ClinicDB, EndoCV2020
from segmentation_models import *
import metrics
from augmentation import *
from torch.utils.data import DataLoader


class ModelEvaluator:
    def __init__(self):
        self.datasets = [
            EtisDataset("Datasets/ETIS-LaribPolypDB"),
            CVC_ClinicDB("Datasets/CVC-ClinicDB"),
            EndoCV2020("Datasets/EndoCV2020"),
        ]
        self.dataloaders = [
                               DataLoader(KvasirSegmentationDataset("Datasets/HyperKvasir", split="test"))] + \
                           [DataLoader(dataset) for dataset in self.datasets]
        self.dataset_names = ["Kvasir-Seg", "Etis-LaribDB", "CVC-ClinicDB", "EndoCV2020"]
        self.models = [DeepLab, FPN, TriUnet, Unet]
        self.model_names = ["DeepLab", "FPN", "TriUnet", "Unet"]

    def get_table_data(self, id_range):

        for model_constructor, model_name in zip(self.models, self.model_names):
            for training_type in ["No_Augmentation", "Augmentation", "Consistency_Training"]:
                mean_ious = np.zeros((len(self.dataloaders), len(id_range)))
                for id in id_range:
                    try:
                        state_fname = f"Predictors/{training_type}_{model_name}-{id}"
                        model = model_constructor().to("cuda")
                        model.load_state_dict(torch.load(state_fname))
                        print(f"Evaluating {state_fname}")
                    except FileNotFoundError:
                        print(f"{state_fname} not found, continuing...")
                        continue

                    for dl_idx, dataloader in enumerate(self.dataloaders):
                        for i, (x, y, _) in enumerate(dataloader):
                            img, mask = x.to("cuda"), y.to("cuda")
                            out = model.predict(img)
                            iou = metrics.iou(out, mask)
                            mean_ious[dl_idx, id - id_range[0]] += iou / len(dataloader)
                print(f"{training_type} {model_name} has a average mIoUs of {np.mean(mean_ious, axis=1 )}")


if __name__ == '__main__':
    evaluator = ModelEvaluator()
    evaluator.get_table_data(range(1,2))