# for the training of the two different types of networks

import os
from utils.dataset import CatsVsDogs, Rescale, ToTensor, EncodeLabel
from utils import NN
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)

    for batch_index, batch in enumerate(dataloader):
        # Extract the data
        (X, y) = batch["image"], batch["label"]

        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        pred = torch.squeeze(pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 10 == 0:
            loss, current = loss.item(), batch_index * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            (X, y) = batch["image"], batch["label"]

            X = X.to(device)
            y = y.to(device)

            test_pred = torch.squeeze(model(X))
            test_loss += loss_fn(test_pred, y).item()
            test_pred = test_pred.round()
            correct += torch.tensor((test_pred == y), dtype=torch.float, device=device).sum()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    """
    If one wants to used colorized images, he should simply put gray attribute of the dataset loading to False and
    adjust the input dimensions of the network in the call accordingly.
    """

    GRAY = True

    if os.name == "nt":
        inPath = "C:\\Users\\Matteo\\Desktop\\Automated Decision Making\\Progetto\\dogs-vs-cats\\train\\train"
    else:
        inPath = "/home/sperimental3/Scrivania/Automated Decision Making/Progetto/dogs-vs-cats/train/train"

    full_dataset = CatsVsDogs(root_dir=inPath,
                              transform=transforms.Compose([Rescale((100, 100)), ToTensor()]),
                              target_transform=EncodeLabel(),
                              gray=GRAY)

    # creation of the dataset for training and testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # print(test_size, train_size)

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    # flag that allow one to train the fully connected network or the CNN (in the latter case one has to put False)
    FC = True

    if FC:
        if GRAY:
            model = NN.Weak(100 * 100 * 1, 1)
        else:
            # colorized case
            model = NN.Weak(100 * 100 * 3, 1)
    else:
        if GRAY:
            model = NN.Strong(1, 1)
        else:
            # colorized case
            model = NN.Strong(3, 1)

    learning_rate = 1e-3

    loss_fn = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    """
        # these operations of "del" can be useful to insert in the training loop in order to use less memory
        for i in range(epochs):
    
            del mini_batch
            del output
    
            torch.cuda.empty_cache()
    """

    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    epochs = 10

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loop(test_dataloader, model, loss_fn, device)

    print("Done!")

    if FC:
        if GRAY:
            torch.save(model.state_dict(), "../tests/FC_gray_weights.pth")
        else:
            torch.save(model.state_dict(), "../tests/FC_color_weights.pth")
    else:
        if GRAY:
            torch.save(model.state_dict(), "../tests/CNN_gray_weights.pth")
        else:
            torch.save(model.state_dict(), "../tests/CNN_color_weights.pth")

    return 0


if __name__ == "__main__":
    main()
