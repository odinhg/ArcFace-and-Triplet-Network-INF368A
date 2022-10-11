import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from os.path import isfile
from tqdm import tqdm
from configfile import *
from utilities import save_embeddings
from dataloader import FlowCamDataLoader
from trainer import train_model
from backbone import BackBone

if __name__ == "__main__":
    if not isfile("./checkpoints/best.pth"):
        exit("No checkpoint found! Please run training before evaluating model.")
    number_of_classes = len(class_names)
    classifier = BackBone(number_of_classes)
    classifier.to(device)
    #Load custom dataset
    train_dataloader, val_dataloader, test_dataloader, _ = FlowCamDataLoader(class_names, image_size, val, test,  batch_size)
    unseen_dataloader = FlowCamDataLoader(class_names_unseen, image_size=image_size, batch_size=batch_size, split=False)
    print("Loading checkpoint.")
    classifier.load_state_dict(torch.load("./checkpoints/best.pth"))
    
    print("Embedding data.")
    save_embeddings(classifier, class_idx, train_dataloader, "embeddings_train.pkl")
    save_embeddings(classifier, class_idx, test_dataloader, "embeddings_test.pkl")
    save_embeddings(classifier, class_idx_unseen, unseen_dataloader, "embeddings_unseen.pkl")
