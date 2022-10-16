import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import FlowCamDataLoader
from backbone import BackBone

torch.manual_seed(0)

# Settings
class_names_all = ["chainthin", "darksphere", "Rhabdonellidae", "Odontella", "Codonellopsis", "Neoceratium", "Retaria", "Thalassionematales", "Chaetoceros"]
class_idx = [0, 1, 2, 3, 4, 5] 
class_idx_unseen = [6, 7, 8]

class_names = [class_names_all[i] for i in class_idx]
class_names_unseen = [class_names_all[i] for i in class_idx_unseen]

number_of_classes = len(class_names)

batch_size = 64
epochs = 50 
lr = 0.0014
val = 0.05 #Use 5% for validation data 
test = 0.2 #Use 20% for test data
image_size = (128, 128)
device = torch.device('cuda:4') 

classifier = BackBone(number_of_classes)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=lr)
