from utils.dataset import CatsVsDogsTest, Rescale, ToTensor
import torch
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import NN
import random
import os


def main():

    # flag to select the type of network and associated MILP to test.
    # For what concern the colorized version of the images the MILP of the CNN is already huge in grayscale,
    # so I think it doesn't make sense from a conceptual point, especially considering that the accuracy
    # in the prediction doesn't change so much given the color,
    # and because the modifications to the model would be almost zero (only the change in the input channel)
    # but the number of variables x_0 would be tripled.

    FC = True
    GRAY = True

    if os.name == "nt":
        inPath = "C:\\Users\\Matteo\\Desktop\\Automated Decision Making\\Progetto\\dogs-vs-cats\\test1\\test1"
    else:
        inPath = "/home/sperimental3/Scrivania/Automated Decision Making/Progetto/dogs-vs-cats/test1/test1"

    test_dataset = CatsVsDogsTest(root_dir=inPath,
                                  transform=transforms.Compose([Rescale((100, 100)), ToTensor()]),
                                  gray=GRAY
                                  )

    dataloader = DataLoader(test_dataset, batch_size=10,
                            shuffle=False, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if FC:
        if GRAY:
            # define net
            net = NN.Weak(1 * 100 * 100, 1)

            # load weights and biases of the previously trained net
            net.load_state_dict(torch.load("FC_gray_weights.pth", map_location=device))
            net = net.eval()
        else:
            # define net
            net = NN.Weak(3 * 100 * 100, 1)

            # load weights and biases of the previously trained net
            net.load_state_dict(torch.load("FC_color_weights.pth", map_location=device))
            net = net.eval()
    else:
        # define net
        net = NN.Strong(1, 1)

        # load weights and biases of the previously trained net
        net.load_state_dict(torch.load("CNN_gray_weights.pth", map_location=device))
        net = net.eval()

    # now let's take some examples and make the network trained classify them, after this we will feed with a real
    # instance (possibly difficult to alter, so an instance in which the prediction of the network is high),
    # then the MILP model will generate an input instance in order to fool the network

    # randomly select from the 5000 images of the test dataset
    rand = random.randint(0, 499)

    for i, sample_batched in enumerate(dataloader):
        if i == rand:
            # print(sample_batched.size())
            # print(sample_batched[3].size())
            # print(torch.unsqueeze(sample_batched[3], dim=0))

            pred = net(sample_batched)
            break

    print(pred)

    pred_max = pred.max()
    pred_min = pred.min()

    if (1 - pred_max) <= pred_min:
        j = pred.argmax()
        pred = pred_max
        sample_label = 1
    else:
        j = pred.argmin()
        pred = pred_min
        sample_label = 0

    index = rand * 10 + j + 1
    print(index, j, sample_label, pred)

    plt.figure()
    grid = utils.make_grid(sample_batched)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()
    plt.show()

    return 0


if __name__ == "__main__":
    main()
"""
Here I will note some good examples of classification:
    FC_gray:
        "801.jpg": 0, pred: 0.1224
        "2991.jpg": 0, pred: 0.1431
        "2903.jpg": 0, pred: 0.1490
        "2590.jpg": 1, pred: 0.7023
        "3809.jpg": 1, pred: 0.6606
        "1053.jpg": 1, pred: 0.6704
        "718.jpg": 1, pred: 0.6458
    FC_color:
        "4150.jpg": 0, pred: 0.2164
        "3496.jpg": 0, pred: 0.2180
        "1179.jpg": 1, pred: 0.8543
        "3016.jpg": 1, pred: 0.8627
        "2306.jpg": 1, pred: 0.9259
    CNN_gray:
        "2609.jpg": 1, pred: 0.9962
        "615.jpg": 0, pred: 0.0084
        "3855.jpg": 1, pred: 0.9766
        "1209.jpg": 0, pred: 0.0048
"""
