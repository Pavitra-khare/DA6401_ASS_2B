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



##  Project File Overview

| **File**       | **Purpose**                                                                 | **Key Functions / Components**                                                                 |
|----------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `run.py`       | Main script to run training & evaluation.                                   | `main()` – Sets up wandb, loads model & data, runs training, logs results.                    |
| `config.py`    | Handles CLI args, device setup, wandb login.                                | Arg parser, device config, `wandb.login()` with API key.                                      |
| `sweep.py`     | Defines hyperparameter sweep settings.                                      | `sweep_config` – Grid search over model strategies, epochs; sets `val_accuracy` metric.       |
| `model.py`     | Loads model & applies transfer learning strategies.                         | `pretrain_model()` – Loads ResNet50, modifies FC, applies one of four freezing strategies.     |
| `data_load.py` | Loads and augments train/val/test data.                                     | `get_transform()`, `data_load()`, `test_data_load()` – Handles dataset splits and transforms. |
| `train.py`     | Training, validation, testing logic with early stopping and wandb logging.  | `train_on_train_data()`, `test_on_valid_data()`, `model_train()`                              |


## File Dependency Map

<pre> 
run.py
├── config.py        # Parses command-line arguments, sets device, logs into W&B  
├── data_load.py     # Loads and augments train, validation, and test datasets  
├── model.py         # Defines CNN model architectures  
├── train.py         # Contains training, validation, testing, and early stopping logic  
└── sweep.py         # (Optional) Contains W&B hyperparameter sweep configurations 
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

---

## How to Run

### 1. Install Requirements

```bash
pip install torch torchvision scikit-learn wandb
```
### 2.Run the file run.py
```bash
python run.py --wandb_project <project_name> --wandb_entity <entity_name> --dpTrain <train_data_path> --dpTest <test_data_path>
```

### Note:- 
GPU enabled enviroment Needed

## Steps to Run on Kaggle

### 1. Upload Python Files  
- Open your Kaggle Notebook.  
- Click on the **"Upload"** button in the file browser pane.  
- Upload the following Python files:  
  - `run.py`  
  - `config.py`  
  - `model.py`  
  - `train.py`  
  - `data_load.py`  
  - `sweep.py`  
- After uploading, click **"Add to notebook"** when prompted.

---

### 2. Upload Dataset  
- Go to the **"Add data"** section on the right-hand panel.  
- Click **"Upload"** and choose your dataset folder from local.  
- Rename the uploaded dataset as **`my_dataset`**.  
- Click **"Add to notebook"** after uploading.

---

### 3. Run the Script  
Use the following command in a code cell to run the main training script with arguments:

```bash
python run.py --wandb_project <project_name> --wandb_entity <entity_name> --dpTrain <train_data_path> --dpTest <test_data_path>
```

