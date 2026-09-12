# RAK-PVM
[Report-Derived Anatomical Knowledge for Pretrained Vision Model Adaptation in Chest X-ray Classification]

![framework](docs/framework.png)

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

### Pre-training and Distillation

**Reminder**: Please check the paths in `pretraining_distillation/dataset/pretrain_dataset.py`, `pretrain_module.py`, and `anatomy_distill.py` and make sure they are correctly configured.

#### Pre-training

We pre-train RAK-PVM on MIMIC-CXR using the following command:
```
cd pretraining_distillation/
CUDA_VISIBLE_DEVICES=0 python pretrain_module.py
```

#### Distillation

After pre-training, we perform report-free knowledge distillation using the following command:
```
cd pretraining_distillation/
CUDA_VISIBLE_DEVICES=0 python anatomy_distill.py
```

Both stages are conducted for 50 epochs on a single NVIDIA RTX PRO 6000 GPU with a batch size of 128. The two stages take approximately one day in total to complete.

### Downstream Classification

**Reminder**: Please check the paths in `downstream/dataloader/factory.py`, `pretrain_module.py`, and `downstream/models/rakpvm.py` and make sure they are correctly configured.

We evaluate the linear classification performance of RAK-PVM using the following command:
```
cd downstream/
CUDA_VISIBLE_DEVICES=0 python main.py
```
The downstream classification stage is conducted for 100 epochs on a single NVIDIA RTX PRO 6000 GPU with a batch size of 128.
