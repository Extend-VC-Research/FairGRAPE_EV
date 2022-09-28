# FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification

This repo presents an official implementation of FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification

![FairGRAPE_concept2](https://user-images.githubusercontent.com/60991389/177890025-4599bd0f-176d-4f5f-aff8-73df9c963a6e.png)

## RE-TRAINING
### Add images to training validation folder
Randomly split the images that are to be added into training and validation sets. Ratio 80:20 or similar. If planning to test with same data batch, then divide as 80:10:10 (test set).  
Name images as _train_XXXXX_ and _val_XXXX_, where XXXX is an ID. Names will be needed for the labelling csv file.

### Labelling images for training/validation
Label tables are stored in the _csv_ folder. For FairFace, open FairFace.csv and fill the relevant columns. **IGNORE** the _service_test_ and _kmeansCluster_ columns as they are not used in the process.
**Alternatively**, create a new csv file with the same format, specific to the new training and validation batch. Then perform merge all the label csv files, to create the input for the training routine.

### Run the training routine
Run in terminal, as per the repo's instructions.
```
python main_test.py --sensitive_group 'race' --loss_type 'race' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'FairFace'  --prune_rate 0.99
```
Note: We could experiment on the training parameters in a later date, and keep an eye on the repo's discussion page, and further publications.

## Dependencies

The code has been tested on the following environment:

```
python 3.9
pytorch 1.11.0
dlib 19.22.0
opencv2 4.5.5
```

Use Anaconda and the following command to replicate the full environment:

```
conda env create -f environment.yml
```

## Datasets

This code automatically downloads the following datasets for trianing, cross validation and testing: [FairFace](https://github.com/joojs/fairface), [UTKFace](https://susanqq.github.io/UTKFace/) and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Request demographics labels of the Imagenet person subtree through the offical [database](https://image-net.org/), save the annotation file under /csv and images under /Images/Imagenet.


## Example Usage:

UTKFace experiments:
```
python main_test.py --sensitive_group 'gender' --loss_type 'race' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'UTKFace'  --prune_rate 0.9 --keep_per_iter 0.975
```

FairFace experiments
```
python main_test.py --sensitive_group 'race' --loss_type 'race' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'FairFace'  --prune_rate 0.99
```

CelebA experiments
```
python main_test.py --sensitive_group 'gender' --loss_type 'classes' --prune_type 'FairGRAPE' --network 'mobilenetv2' --dataset 'CelebA' --prune_rate 0.9
```

Imagenet experiments:
```
python main_test.py --sensitive_group 'gender' --loss_type 'classes' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'Imagenet'  --prune_rate 0.5
```

Download trained models [here](https://www.dropbox.com/sh/rk362mypuikeklh/AADF93dWPQo3rPTUhyaLBn3Ga?dl=0).

Loading a trained model:
```
python main_test.py --sensitive_group 'gender' --loss_type 'race' --prune_type 'FairGRAPE' --network 'mobilenetv2'--dataset 'UTKFace'  --prune_rate 0.9 --keep_per_iter 0.975 --checkpoint "UTKFace_FairGRAPE_race_bygender_resnet34_0.9_0.pt" --init_train 0 --retrain 0 --print_acc 1
```


## Acknowledgement 
Parts of code were borrowed from [SNIP](https://github.com/mil-ad/snip), [WS](https://github.com/mightydeveloper/Deep-Compression-PyTorch), [GraSP](https://github.com/alecwangcq/GraSP)
