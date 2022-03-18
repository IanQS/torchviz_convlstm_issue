import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from Seq2Seq import Seq2Seq
from torch.utils.data import DataLoader

import io
import imageio
from ipywidgets import widgets, HBox

# Use GPU if available
from torchviz import make_dot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MovingMNIST = np.load('mnist_test_seq.npy').transpose(1, 0, 2, 3)

# Shuffle Data
np.random.shuffle(MovingMNIST)

# Train, Test, Validation splits
train_data = MovingMNIST[:8000]
val_data = MovingMNIST[8000:9000]
test_data = MovingMNIST[9000:10000]

def collate(batch):

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)

    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(10,20)
    return batch[:,:,rand-10:rand], batch[:,:,rand]


# Training Data Loader
train_loader = DataLoader(train_data, shuffle=True,
                          batch_size=16, collate_fn=collate)

# Validation Data Loader
val_loader = DataLoader(val_data, shuffle=True,
                        batch_size=16, collate_fn=collate)

if __name__ == '__main__':

    model = Seq2Seq(num_channels=1, num_kernels=64,
                    kernel_size=(3, 3), padding=(1, 1), activation="relu",
                    frame_size=(64, 64), num_layers=3).to(device)

    optim = Adam(model.parameters(), lr=1e-4)

    # Binary Cross Entropy, target pixel values either 0 or 1
    criterion = nn.BCELoss(reduction='sum')
    num_epochs = 2

    for epoch in range(1, num_epochs+1):

        train_loss = 0
        model.train()
        for batch_num, (input, target) in enumerate(train_loader, 1):
            output = model(input)
            bla = make_dot(
                output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True
            )
            bla.render("/tmp/model_y_hat_ob", format="png")

            loss = criterion(output.flatten(), target.flatten())
            loss.backward()
            optim.step()
            optim.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for input, target in val_loader:
                output = model(input)
                loss = criterion(output.flatten(), target.flatten())
                val_loss += loss.item()
        val_loss /= len(val_loader.dataset)

        print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
            epoch, train_loss, val_loss))
