"""
Contains the functionality for the creating Pytorch Dataloaders for the image classification task.
"""
# Import the some important libraries
import os
import torch
from torchvision import datasets , transforms
from torch.utils.data import Dataloader
from typing import Tuple , List , Dict
from tqdm import  tqdm
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from pathlib import Path

NUM_WORKERS = os.cpu_count()
def create_dataloaders(
        train_dir:str,
        test_dir:str,
        transform:transforms.Compose,
        batch_size:int = 32,
        num_workers:int = NUM_WORKERS
):
    """ 
    Creates training and testing Dataloaders

    Takes in a training directory and testing directory path and turns then into PyTorch Datasets and then into PyTorch Dataloaders.

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transform: torchvision transforms t perform on training and testing data
        batch_size: Number of samples per batch in each of the Dataloaders
        num_workers: An integer for the number of workers per Dataloader
    
    Returns:
        A tuple of (train_dataloader,test_dataloader , class_names).
        Where class_names is a list of the target classes.
        Example usage:
        train_dataloader, test_dataloader, class_names = create_dataloaders(
            train_dir = "path/to/train_dir",
            test_dir = "path/to/test_dir",
            transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]),
            batch_size = 32,
            num_workers = 4
        )
    """
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir , transform=transform)
    test_data = datasets.ImageFolder(test_dir , transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = Dataloader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = Dataloader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Now return the train_dataloader , test_dataloader , class_names
    return train_dataloader , test_dataloader , class_names

'''Contains the functions for training and testing the model'''
def train_step(
        model:torch.nn.Module,
        dataloader:torch.utils.data.Dataloader,
        loss_fn:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        device:torch.device
) -> Tuple[float , float]:
    """Trains a Pytorch model for a single epoch.

    Turns a target PyTorch model to training mode and then runs through all of the required training steps(forward
    pass,loss calculation , optimizer step).

    Args:
        model:A PyTorch model to be trained.
        dataloader:A Dataloader instance for the model to be trained on.
        loss_fn:A loss function to be used for the training.
        optimizer:A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss and training accuracy metrics. In the form of (train_loss , train_acc).For example:(0.1112 , 0.8743).
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss , train_acc = 0 , 0

    # Loop through data loader data batches
    for batch , (X,y) in enumerate(dataloader):
        # Send data to target device
        X , y = X.to(device) , y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred , y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward pass
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred , dim=1) , dim=1)
        train_acc += torch.sum(y_pred_class == y).sum().item()/len(y_pred_class)
    
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss , train_acc

# Write code for prepare the function to test the model
def test_step(model:torch.nn.Module,dataloader:torch.utils.data.Dataloader,loss_fn:torch.nn.Module,device:torch.device) -> Tuple[float,float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to evaluation mode and then runs through all of the required testing steps(forward
    pass,loss calculation).

    Args:
        model:A PyTorch model to be tested.
        dataloader:A Dataloader instance for the model to be tested on.
        loss_fn:A loss function to be used for the testing.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics. In the form of (test_loss , test_acc).For example:(0.1112 , 0.8743).
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss , test_acc = 0 , 0

    # Turn on inference context manager
    with torch.inference_mode():
        for batch , (X,y) in enumerate(dataloader):
            # Send data to target device
            X , y = X.to(device) , y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits , y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy metric across all batches
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += torch.sum(test_pred_labels == y).sum().item()/len(test_pred_labels)
    
    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss , test_acc

# Now we will write the function to train and test the model
def train(
        model:torch.nn.Module,
        train_dataloader:torch.utils.data.Dataloader,
        test_dataloader:torch.utils.data.Dataloader,
        optimizer:torch.optim.Optimizer,
        loss_fn:torch.nn.Module,
        epochs:int,
        device:torch.device
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.
    
    Passes a target PyTorch models through a train_step() and test_step() functions for a number of epochs,training and testing the model
    in the same epoch loop.

    Calculates , prints and stores evaluation metrics throughout.

    Args:
        model:A PyTorch model to be trained and tested.
        train_dataloader:A Dataloader instance for the model to be trained on.
        test_dataloader:A Dataloader instance for the model to be tested on.
        optimizer:A PyTorch optimizer to help minimize the loss function.
        loss_fn:A loss function to be used for the training and testing.
        epochs:An integer for the number of epochs to train and test the model for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
            A dictionary of training and testing loss as well as training and testing accuracy metrics. Each metric is a list of values in a list for 
            each epoch.
            In the form: {"train_loss":[0.1112,0.1234],"train_acc":[0.8743,0.8765],"test_loss":[0.1112,0.1234],"test_acc":[0.8743,0.8765]}
            For example if training for epochs=2:
            {
                "train_loss":[0.1112,0.1234],
                "train_acc":[0.8743,0.8765],
                "test_loss":[0.1112,0.1234],
                "test_acc":[0.8743,0.8765]
            }
    """
    # Create empty results dictionary
    results = {
        "train_loss":[],
        "train_acc":[],
        "test_loss":[],
        "test_acc":[],
    }

    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss , train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss , test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        # Print out what's happening
        print(f"Epoch: {epoch+1}/{epochs} | "
              f"Train loss: {train_loss:.4f} | "
              f"Train accuracy: {train_acc:.4f} | "
              f"Test loss: {test_loss:.4f} | "
              f"Test accuracy: {test_acc:.4f}")
        
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results

# Function to plot the results of the training and testing
def pred_and_plot_image(
        model:torch.nn.Module,
        class_names:List[str],
        image_path:str,
        image_size:Tuple[int,int]=(224,224),
        transform: torchvision.transforms = None,
        device:torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """Predicts on a target image with a target model.
    
    Args:
        model:A PyTorch model to be used for prediction.
        class_names:A list of class names for the target dataset.
        image_path:A path to a target image to be predicted on.
        image_size: A tuple of integers for the target image size.
        transform:A torchvision transforms to perform on the image.
        device(torch.device , optional): A target device to compute on (e.g. "cuda" or "cpu").
    """
    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406], std=[0.229 , 0.224 , 0.225]
            )
        ])
    
    ## Predict on image
    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Trasform and add an extra dimension to the image (model requires samples in [batch_size , colours_channels , height , width])
        transformed_image = image_transform(img).unsqueeze(dim=0)
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities(using torch.softmax() for multiclass classification)
    target_image_pred_probs = torch.softmax(target_image_pred , dim=1)
    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs , dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)


"""Contains various utility functions for PyTorch model training and saving"""
def save_model(
        model:torch.nn.Module,
        target_dir:str,
        model_name:str
):
    """Saves a PyTorch model to a target directory.
    
    Args:
        model:A PyTorch model to be saved.
        target_dir:A directory to save the model to.
        model_name:A filenname for the saved model. Should include either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(
            model=model,
            target_dir="path/to/save/model",
            model_name="model.pth"
        )
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True , exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "Please provide a .pth or .pt file extension"
    model_save_path = target_dir_path / model_name

    # Save model state_dict()
    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
