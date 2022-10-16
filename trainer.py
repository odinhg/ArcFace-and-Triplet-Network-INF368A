import torch
from tqdm import tqdm
import numpy as np
from utilities import EarlyStopper
from os.path import join
from configfile import *

def train_model(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device):
    train_history = {"train_loss":[], "train_accuracy":[], "val_loss":[], "val_accuracy":[]}
    steps = len(train_dataloader) // 5 #Compute validation and train loss 5 times every epoch
    earlystop = EarlyStopper()
    for epoch in range(epochs):
        train_losses = []
        train_accuracies = []
        for i, data in enumerate((pbar := tqdm(train_dataloader))):
            images, labels  = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracies.append((predicted == labels).sum().item() / labels.size(0))
            if i % steps == steps - 1:
                train_loss = np.mean(train_losses)
                train_losses = []
                train_accuracy = 100.0 * np.mean(train_accuracies)
                correct = 0
                total = 0
                val_losses = []
                classifier.eval()
                with torch.no_grad():
                    for data in val_dataloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = classifier(images)
                        val_losses.append(loss_function(outputs, labels).item())
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    val_accuracy = 100.0 * correct / total
                    val_loss = np.mean(val_losses)
                if train_history["val_accuracy"] and val_accuracy > np.max(train_history["val_accuracy"]):
                    torch.save(classifier.state_dict(), join(checkpoints_path, "best.pth"))
                train_history["train_loss"].append(train_loss)
                train_history["train_accuracy"].append(train_accuracy)
                train_history["val_loss"].append(val_loss)
                train_history["val_accuracy"].append(val_accuracy)
                pbar.set_description(f"Epoch {epoch}/{epochs-1} | Loss: Train={round(train_loss, 3)} Val={round(val_loss, 3)} | Acc.: Train={round(train_accuracy,1)}% Val={round(val_accuracy, 1)}%")
                if earlystop(val_loss):
                    print(f"Early stopped at epoch {epoch}")
                    return train_history
                classifier.train()
    return train_history
