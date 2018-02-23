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

## Training ensemble from scratch
This section describes the steps required to train our ensemble.
    
### 1. Download external dataset
Images from both Yandex.Fotki and Flickr are essential for reproducing our solution.

#### Downloading images from Yandex.Fotki
Run `bash download_from_yandex.sh`

#### Downloading images from Flickr
Unfortunately, this step involves some manual actions.
1. cd into `downloader/flickr`
2. For every model go to the telephone model group page from `flickr_groups.txt`. Scroll every gallery
page to the end and download as html file to the corresponding folder. As a result you
will have a set of folders with .html files corresponding to a specific phone model
at `html_pages` folder.
3. Run `python pages_to_image_links.py`. The result of the script will be
folder `links` of .csv files with links to photos of each phone model.
4. Run `python download_from_links.py` to download images from the links received in
the previous paragraph (previous two steps could be skipped, because the
`links` folder already contains necessary files).

### 2. Filter external dataset
Run `bash filter.sh`

### 3. Train the ensemble
1. Download and filter external dataset as described above.
2. Run `bash init_train.sh` to train 9 models.
3. Run `bash make_pseudo.sh` to get predictions from these models for images at `data/test` and 
create pseudo labels.
4. Run `bash final_train.sh` to train the same 9 models but using pseudo labels this time.
5. Run `bash predict.sh -d <folder with test images> -o <output .csv filename>` to get
predictions from the ensemble.