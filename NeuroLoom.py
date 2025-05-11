"""
Contains the functionality for the creating Pytorch Dataloaders for the image classification task.
"""
# Import the some important libraries
import os
import torch
from torchvision import datasets , transforms
from torch.utils.data import DataLoader
from typing import Tuple , List , Dict
from tqdm import  tqdm
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from pathlib import Path
import requests
import zipfile
import time

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
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
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
        dataloader:torch.utils.data.DataLoader,
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
def test_step(model:torch.nn.Module,dataloader:torch.utils.data.DataLoader,loss_fn:torch.nn.Module,device:torch.device) -> Tuple[float,float]:
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
        train_dataloader:torch.utils.data.DataLoader,
        test_dataloader:torch.utils.data.DataLoader,
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

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Pred and plot image function 
def pred_and_plot_images(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path
