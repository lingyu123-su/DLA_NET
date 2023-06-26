import torchvision
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        self.sketch_branch = self._get_resnet50()
        self.photo_branch = self._get_resnet50()

    def _get_resnet50(self):
        resnet50_model = torchvision.models.resnet50(pretrained=True)
        # Remove the GAP layer and following layers
        modules=list(resnet50_model.children())[:-3]
        # Remove the final convolution layer and get the output of conv3 layer
        modules = modules[:-1]
        return nn.Sequential(*modules)

    def forward(self, sketch_input, photo_input):
        sketch_output = self.sketch_branch(sketch_input)
        photo_output = self.photo_branch(photo_input)
        return sketch_output, photo_output

class TripletDataset(Dataset):
    def __init__(self, sketch_dir, photo_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.photo_dir = photo_dir
        self.transform = transform

        # Get list of all files in sketch_dir and photo_dir
        self.sketch_files = os.listdir(sketch_dir)
        self.photo_files = os.listdir(photo_dir)

    def __len__(self):
        return min(len(self.sketch_files), len(self.photo_files))

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.sketch_files[idx])
        positive_photo_path = os.path.join(self.photo_dir, self.photo_files[idx])

        # For simplicity, the negative example is just another random photo image
        negative_idx = random.randint(0, len(self.photo_files) - 1)
        while negative_idx == idx:  # Ensure the negative example is different from the positive example
            negative_idx = random.randint(0, len(self.photo_files) - 1)
        negative_photo_path = os.path.join(self.photo_dir, self.photo_files[negative_idx])

        sketch_image = Image.open(sketch_path).convert('RGB')
        positive_image = Image.open(positive_photo_path).convert('RGB')
        negative_image = Image.open(negative_photo_path).convert('RGB')

        if self.transform:
            sketch_image = self.transform(sketch_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return sketch_image, positive_image, negative_image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = TripletDataset(sketch_dir='./sketches', photo_dir='./photos', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)


model = TripletNet().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.TripletMarginLoss(margin=1.0, p=2)

for epoch in range(num_epochs):
    for i, (sketch_images, positive_images, negative_images) in enumerate(dataloader):
        sketch_images = sketch_images.to(device)
        positive_images = positive_images.to(device)
        negative_images = negative_images.to(device)

        sketch_output, positive_output = model(sketch_images, positive_images)
        _, negative_output = model(None, negative_images)

        # Compute the loss
        loss = criterion(sketch_output, positive_output, negative_output)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ...
