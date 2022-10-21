import torch
import numpy as np
import pandas as pd
from os.path import isfile, join
from tqdm import tqdm
from configfile import *
from utilities import save_train_plot
from dataloader import FlowCamDataLoader
from trainer import train_classifier, train_triplet
from torchsummary import summary

if __name__ == "__main__":
    #Use custom backbone based on EfficientNet v2
    summary(classifier, (3, *image_size), device=device)
    classifier.to(device)

    if not isfile(join(checkpoints_path, "best.pth")):
        print("Training...")
        if model_type == "triplet":
            train_history = train_triplet(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        else:
            train_history = train_classifier(classifier, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
            save_train_plot(join(figs_path, "training_plot.png"), train_history)
    else:
        print("Chechpoint found! Please delete checkpoint and run training again.")
