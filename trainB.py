import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50, ResNet50_Weights
import wandb




# ---------------------------------------------------------------------------------------

# --- Sweep configuration for hyperparameter search ---
def get_sweep_config(args):
    return {
        'method': 'grid',  # Use grid search to test all strategies
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'model': {'values': ['resnet']},
            'strategy': {
                'values': [
                    'freeze_all_except_last',  # Only last layer trainable
                    'freeze_fc_only',         # Only FC layer frozen
                    'freeze_80_percent',      # Freeze 80% of layers
                    'train_from_scratch'      # No pretraining, all trainable
                ]
            },
            'epoch': {'values': [args.epoch]}
        }
    }


# ---------------------------------------------------------------------------------------

# --- Argument parsing and device setup ---
def parse_and_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", default="DL_ASS2")  # W&B project name
    parser.add_argument("-we", "--wandb_entity", default="3628-pavitrakhare-indian-institute-of-technology-madras")  # W&B entity
    parser.add_argument("-key", "--wandb_key", default="bad0d13cb33ad3ab10579145135ecdce4cd371f0")  # W&B API key
    parser.add_argument("-dpTrain", "--dpTrain", default="/kaggle/input/my-dataset/inaturalist_12K/train")  # Train data path
    parser.add_argument("-dpTest", "--dpTest", default="/kaggle/input/my-dataset/inaturalist_12K/val")      # Test data path
    parser.add_argument("-ep","--epoch", type=int, default=10)    # no of epochs
    args = parser.parse_args()
    wandb.login(key=args.wandb_key)  # Authenticate with W&B
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    print(device)
    return args, device



# ---------------------------------------------------------------------------------------

# --- Data transformation pipeline ---
def get_transform(augment):
    # Return data augmentation pipeline if requested
    if augment == 'Yes':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    # Standard resize and normalization for validation/test
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])



# ---------------------------------------------------------------------------------------

# --- Training and validation data loader ---
def loadTheData(data_dir, data_augumentation):
    # Apply chosen transform pipeline
    transform_pipeline = get_transform(data_augumentation)
    # Load entire dataset with transforms
    complete_dataset = ImageFolder(root=data_dir, transform=transform_pipeline)
    dataset_size = len(complete_dataset)
    indices = list(range(dataset_size))
    # Split indices for 80% train, 20% validation
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    # DataLoader for training set
    train_loader = DataLoader(
        complete_dataset, batch_size=32, sampler=train_sampler, num_workers=4, pin_memory=True
    )
    # DataLoader for validation set
    val_loader = DataLoader(
        complete_dataset, batch_size=32, sampler=val_sampler, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader



# ---------------------------------------------------------------------------------------

# --- Test data loader ---
def LoadTheTestData(test_dir, data_augumentation='No'):
    # Apply transform and load test set
    transform_pipeline = get_transform(data_augumentation)
    dataset = ImageFolder(root=test_dir, transform=transform_pipeline)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    return loader



# ---------------------------------------------------------------------------------------

# --- Model setup and strategy selection ---
def modelStratPretrain(model_name, strategy):
    # Use pretrained weights unless training from scratch
    use_pretrained = (strategy != 'train_from_scratch')
    pretrained_model = load_model(model_name, use_pretrained)
    num_features = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_features, 10)  # Set output for 10 classes
    # Apply freezing strategy
    if strategy == 'freeze_all_except_last':
        freeze_except_fc(pretrained_model)
    elif strategy == 'freeze_80_percent':
        freeze_initial_80_percent(pretrained_model)
    elif strategy == 'freeze_fc_only':
        freeze_fc_layer(pretrained_model)
    # 'train_from_scratch': all layers trainable by default
    return pretrained_model



# ---------------------------------------------------------------------------------------

def load_model(name, use_pretrained):
    # Load ResNet50 with or without pretrained weights
    if name == 'resnet':
        model_weights = ResNet50_Weights.DEFAULT if use_pretrained else None
        return resnet50(weights=model_weights)
    else:
        raise ValueError(f"Unsupported model: {name}")
    


# ---------------------------------------------------------------------------------------

def freeze_except_fc(model):
    # Freeze all layers except the final fully connected layer
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False



# ---------------------------------------------------------------------------------------

def freeze_fc_layer(model):
    # Freeze only the fully connected layer
    for name, param in model.named_parameters():
        if name.startswith("fc"):
            param.requires_grad = False



# ---------------------------------------------------------------------------------------

def freeze_initial_80_percent(model):
    # Freeze the first 80% of model layers
    children = list(model.named_children())
    freeze_upto = int(len(children) * 0.8)
    for idx, (_, module) in enumerate(children):
        if idx < freeze_upto:
            for param in module.parameters():
                param.requires_grad = False



# ---------------------------------------------------------------------------------------

# --- Training loop for one epoch ---
def trainDataTraining(model, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()  # Set model to training mode
    epoch_loss = 0
    total_correct = 0
    total_count = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients
        output = model(images)  # Forward pass
        loss = criterion(output, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        epoch_loss += loss.item()
        preds = output.argmax(dim=1)  # Get predicted class
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)
    avg_loss = epoch_loss / len(train_loader)
    accuracy = (total_correct / total_count) * 100
    return model, avg_loss, accuracy



# ---------------------------------------------------------------------------------------

# --- Validation/testing loop ---
def validDataTesting(model, test_data, device):
    model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():  # No gradients needed during evaluation
        for inputs, targets in test_data:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted_classes = torch.max(outputs, dim=1)
            correct_predictions += (predicted_classes == targets).sum().item()
            total_samples += targets.size(0)
    accuracy = (correct_predictions / total_samples) * 100
    return accuracy



# ---------------------------------------------------------------------------------------

# --- Full model training and evaluation ---
def trainCnnModel(model, train_data, val_data, test_data, epochs, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    highest_val_acc = 0  # Track best validation accuracy
    stop_limit = 2       # Early stopping patience
    stop_counter = 0
    for ep in range(epochs):
        model, loss_avg, acc_train = trainDataTraining(model, train_data, device)
        print(f'|| Epoch {ep + 1}/{epochs} ')
        print(f'Training Accuracy: {acc_train:.1f}% ')
        print(f' Training Loss: {loss_avg:.3f}')
        wandb.log({'Train loss': loss_avg})
        wandb.log({'Train accuracy': acc_train})
        acc_val = validDataTesting(model, val_data, device)
        print(f'Validation Accuracy: {acc_val:.1f}% ||')
        print()
        wandb.log({'val_accuracy': acc_val})
        wandb.log({'epoch': ep})
        # Early stopping logic: stop if no improvement
        if acc_val > highest_val_acc:
            highest_val_acc = acc_val
            stop_counter = 0
        else:
            stop_counter += 1
        if stop_counter >= stop_limit:
            print(f"Early stopping at epoch {ep + 1}")
            break
    # Final evaluation on test data
    acc_test = validDataTesting(model, test_data, device)
    print(f'Test Accuracy: {acc_test:.2f}%')
    wandb.log({'test_accuracy': acc_test})
    print('Model training Completed.')


# ---------------------------------------------------------------------------------------

# --- Main execution: orchestrates the full experiment ---
def main():

    args, device = parse_and_setup()  # Parse arguments and set device
    sweep_config = get_sweep_config(args) # Get sweep configuration for W&B

    def sweep_main():
        with wandb.init() as run:
            config = wandb.config  # Access current hyperparameters
            # Create a descriptive run name for W&B
            run_name_parts = [
                f"ep{config.epoch}",
                f"strategy-{config.strategy}",
                f"model-{config.model}"
            ]
            wandb.run.name = "_".join(run_name_parts)
            # Initialize model with selected strategy
            selected_model = modelStratPretrain(model_name=config.model, strategy=config.strategy)
            selected_model = selected_model.to(device)
            # Load training and validation data
            train_data, val_data = loadTheData(args.dpTrain, data_augumentation='No')
            # Load test data
            test_data = LoadTheTestData(args.dpTest, data_augumentation='No')
            # Train and evaluate the model
            trainCnnModel(selected_model, train_data, val_data, test_data, epochs=config.epoch, device=device)

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project,entity=args.wandb_entity)  # Register sweep with W&B
    wandb.agent(sweep_id, function=sweep_main, count=4)           # Run sweep agent for all strategies
    wandb.finish()  # Finish W&B run


# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
