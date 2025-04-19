# DL_Assignment2 - CS24M031 

I used kaggle for this assignment. First create a kaggle account and go to the code section and make new notebook there and upload the data set and copy to the path to access the train and test datasets.

Here we are implementing a CNN model on the iNaturalist dataset from scratch and tuned the hyperparameters to achieve optimal performance. I utilized Python along with required packages from PyTorch and Torchvision.

[GitHub Repository](https://github.com/Pavitra-khare/DA6401_ASS_2B/tree/main)  
[Weights & Biases Report](https://api.wandb.ai/links/3628-pavitrakhare-indian-institute-of-technology-madras/m5cmjze4)

## PART B

---

### Model Architecture
**ResNet-50 Implementation**  
| Feature               | Specification                          |
|-----------------------|----------------------------------------|
| Base Model            | ResNet50 (torchvision implementation) |
| Depth                 | 50 layers                              |
| Input Size            | 224x224 RGB images                     |
| Output Layer          | Custom FC layer (10 classes)           |
| Pretrained Weights    | ImageNet (when not training from scratch) |


---





## Functions structure

<pre> 
## Single-File Implementation (trainB.py)

### Core Function Breakdown

| **Function**                | **Purpose**                                                                 | **Key Parameters**                          | **Returns**                              |
|-----------------------------|-----------------------------------------------------------------------------|---------------------------------------------|------------------------------------------|
| `get_sweep_config()`         | Configures W&B hyperparameter search strategies                             | -                                           | Dictionary of sweep parameters           |
| `parse_and_setup()`          | Parses CLI arguments & initializes hardware/W&B                             | `--wandb_project`, `--wandb_entity`, etc.   | (args, device) tuple                     |
| `get_transform(augment)`     | Creates image transformation pipelines                                      | `augment`: Yes/No for augmentation         | `transforms.Compose` object              |
| `loadTheData(data_dir, ...)` | Loads training/validation data with split                                   | `data_dir`: Dataset path                    | (train_loader, val_loader) tuple         |
| `modelStratPretrain(...)`     | Initializes ResNet50 with transfer learning strategies                      | `strategy`: Freezing approach               | Configured PyTorch model                 |
| `trainDataTraining()`        | Executes one training epoch                                                 | `model`, `train_loader`, `device`           | Updated model, loss, accuracy            |
| `validDataTesting()`         | Evaluates model on validation/test data                                     | `model`, `test_data`, `device`              | Accuracy percentage                      |
| `trainCnnModel()`            | Orchestrates full training loop with early stopping                         | `model`, train/val/test data, `epochs`      | Final trained model                      |

---


</pre>

---

## Key Results
| Metric | Value | Configuration |
|--------|-------|---------------|
| Best Val Accuracy | 80.65% | Freeze_all_except_last |
| Test Accuracy | 81.20% | 10 epochs |


---

## Command Line Arguments
| Argument | Default Value | Description |
|----------|---------------|-------------|
| `-wp/--wandb_project` | DL_ASS2 | Weights & Biases project name |
| `-we/--wandb_entity` | 3628-pavitrakhare-indian-institute-of-technology-madras| W&B team/username |
| `-key/--wandb_key` | [hidden] | W&B API authentication key |
| `-dpTrain/--dpTrain` | /kaggle/input/my-dataset/inaturalist_12K/train| Training dataset path |
| `-dpTest/--dpTest` | /kaggle/input/my-dataset/inaturalist_12K/val | Test dataset path |
| `-ep/--epoch` | 10 | number of Epochs you want to run |


---

## How to Run

### 1. Install Requirements

```bash
pip install torch torchvision scikit-learn wandb
```
### 2.Run the file trainB.py
```bash
python trainB.py --wandb_project <project_name> --wandb_entity <entity_name> --dpTrain <train_data_path> --dpTest <test_data_path>
```

### Note:- 
GPU enabled enviroment Needed

