# Settings triplet loss contrastive learning
#batch_size = 64
#batch_size = 128
batch_size = 64 
epochs = 10 
lr = 0.0014
val = 0.05 #Use 5% for validation data 
test = 0.2 #Use 20% for test data
image_size = (128, 128)

margin = 0.2 # Triplet loss margin

model_type = "triplet"
