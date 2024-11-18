import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import InceptionResnetV1
import cv2

# Define a simple Siamese Network architecture
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_model):
        super(SiameseNetwork, self).__init__()
        self.embedding_model = embedding_model  # Pre-trained face recognition model

    def forward(self, x1, x2):
        # Get embeddings for both images
        embedding1 = self.embedding_model(x1)
        embedding2 = self.embedding_model(x2)
        return embedding1, embedding2

# Contrastive loss for learning similarity
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2))
        loss = label * torch.pow(euclidean_distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return loss.mean()

# Sample dataset (pairs of images with similarity labels)
class FaceVerificationDataset(Dataset):
    def __init__(self, image_pairs, labels, transform=None):
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        label = self.labels[idx]
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

# Example usage of the Siamese Network
def train_siamese_network(train_loader, model, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for data1, data2, labels in train_loader:
            optimizer.zero_grad()

            # Get embeddings
            embedding1, embedding2 = model(data1, data2)

            # Calculate loss
            loss = criterion(embedding1, embedding2, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Initialize the pre-trained model, Siamese Network, and other components
embedding_model = InceptionResnetV1(pretrained='vggface2').eval()  # Pretrained face model
siamese_model = SiameseNetwork(embedding_model)
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(siamese_model.parameters(), lr=0.0001)

# Create sample data and DataLoader
image_pairs = [('path_to_face1.jpg', 'path_to_face2.jpg'), ('path_to_face3.jpg', 'path_to_face4.jpg')]  # Example pairs
labels = [1, 0]  # 1: Same person, 0: Different person
train_dataset = FaceVerificationDataset(image_pairs, labels)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Train the model
train_siamese_network(train_loader, siamese_model, criterion, optimizer)
