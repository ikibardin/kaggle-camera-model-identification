# Kaggle IEEE's Signal Processing Society - Camera Model Identification

Implementation of camera model identification system by team "[ods.ai] GPU_muscles" (2nd place overall in Kaggle
competition "IEEE's Signal Processing Society - Camera Model Identification" and 1st place among student eligible
teams).

## Requirements
To train models and get predictions the following is required:

- OS: Ubuntu 16.04
- Python 3.5
- Hardware:
    - Any decent modern computer with x86-64 CPU, 
    - 32 GB RAM
    - 4 x Nvidia GeForce GTX 1080 Ti

## Installation
1. Install required OS and Python
2. Install packages with `pip install -r requirements.txt`
3. Create `data` folder at the root of the repository. Place train dataset from Kaggle 
competition to `data/train`. Place test dataset from Kaggle competition to `data/test`. 
Place additional validation images to `data/val_images`.
4. Place `se_resnet50.pth` and `se_resnext50.pth` to `imagenet_pretrain` folder.
5. Place the following final weights to `final_weights` folder:
    - `densenet161_28_0.08377413648371115.pth`
    - `densenet161_55_0.08159203971706519.pth`
    - `densenet161_45_0.0813179751742137.pth`
    - `dpn92_tune_11_0.1398952918197271.pth`
    - `dpn92_tune_23_0.12260739478774665.pth`
    - `dpn92_tune_29_0.14363511492280367.pth`
    
## Producing the final submission
Run `bash final_submit.sh -d <folder with test images> -o <output .csv filename>`
    
## Downloading external dataset
???

## Filtering external dataset
???

## Training the ensemble
1. Download and filter external dataset as described above.
2. Run `bash init_train.sh` to train 9 models.
3. Run `bash make_pseudo.sh` to get predictions from these models for images at `data/test` and 
create pseudo labels.
4. Run `bash final_train.sh` to train the same 9 models but using pseudo labels this time.
5. Run `bash predict.sh -d <folder with test images> -o <output .csv filename>` to get
predictions from the ensemble.