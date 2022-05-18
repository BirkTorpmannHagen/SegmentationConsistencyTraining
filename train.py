from trainers import *
import sys

if __name__ == '__main__':
    model = sys.argv[1]
    id = sys.argv[2]
    config = {"model": model,
              "device": "cuda",
              "lr": 0.00001,
              "batch_size": 8,
              "epochs": 300,
              "use_inpainter": False}


    """
    Consistency Training
    """
    trainer = ConsistencyTrainer(id, config)
    trainer.train()
    """
    Model-based augmentations
    """
    trainer = AugmentationTrainer(id, config)
    trainer.train()
    """
        No augmentations
    """
    trainer = VanillaTrainer(id, config)
    trainer.train()
