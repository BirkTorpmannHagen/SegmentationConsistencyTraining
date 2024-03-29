
# Segmentation Consistency Training:Out-of-Distribution Generalization for Medical Image Segmentation

This repository is the official implementation of _Segmentation Consistency Training:Out-of-Distribution Generalization for Medical Image Segmentation_. 
The paper is available [here](https://www.computer.org/csdl/proceedings-article/ism/2022/717200a042/1KaHM68T1Ty)

![method](https://github.com/BirkTorpmannHagen/SegmentationConsistencyTraining/blob/master/consistency_training.png)
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train an instance of a predictor with a given model architecture and an ID number (to name the saved models), run:

```train
python train.py [Model Architecture] [ID]
```
To train ten predictors for each model architecture, run:
```trainall
bash run_paper_experiments.sh
```

## Evaluation

To evaluate the models as trained using the above script, run:

```eval
python eval.py [ID lower] [ID upper]
```
This will evaluate the models given with IDs in the range [ID lower, ID ipper]

## Results
![improvements](https://github.com/BirkTorpmannHagen/SegmentationConsistencyTraining/blob/master/consistency_training_percent.png)

We demonstrate that Segmentation Inconsistency Training improves generaliation by a statistically significant margin (p>0.99) on all three tested out-of-distribution datasets.

##Contact information
For questions or help, feel free to contact me at birk.s.torpmann-hagen@uit.no.

