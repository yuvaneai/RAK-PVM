# RAK-PVM
[Report-Derived Anatomical Knowledge for Pretrained Vision Model Adaptation in Chest X-ray Classification]

![framework](docs/framework.pdf)

###  Installation
To clone this repository:
```
git clone https://github.com/yuvaneai/RAK-PVM.git
```
To install Python dependencies:
```
pip install -r requirements.txt
```

### Dataset downloading

Datasets we used are as follows:

* **MIMIC-CXR**: We downloaded the [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) dataset as the radiographs. Paired medical reports can be downloaded in [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/mimic-cxr-reports.zip).

* **ChestX-Det10**: We downloaded the [ChestX-Det10](https://github.com/wangtao123456/ChestX-Det10) dataset as the chest radiographs.

* **NIH ChestX-ray14**: We downloaded the [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) dataset as the chest radiographs.

### Data Preprocessing
We preprocessed these datasets and split the dataset into train/test set using the code in `datapreprocess`.

