from utils.dataset import CatsVsDogsTest, Rescale, ToTensor
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import torch
from core import MILP, MILP_extreme
import os


def main():

    FC = True
    GRAY = True

    if GRAY:
        label = "gray"
    else:
        label = "color"

    if os.name == "nt":
        inPath = "C:\\Users\\Matteo\\Desktop\\Automated Decision Making\\Progetto\\dogs-vs-cats\\test1\\test1"
    else:
        inPath = "/home/sperimental3/Scrivania/Automated Decision Making/Progetto/dogs-vs-cats/test1/test1"

    test_dataset = CatsVsDogsTest(root_dir=inPath,
                                  transform=transforms.Compose([Rescale((100, 100)), ToTensor()]),
                                  gray=GRAY
                                  )

    number = 801
    # here I've put number - 1 because even if the images start from 1,
    # the dataset and the dataloader start to count from 0
    sample = test_dataset[number - 1]
    sample_label = 0

    # print(sample.shape)

    # plt.imshow(sample.numpy().transpose((1, 2, 0)), cmap="gray")
    # plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # read weights and biases from file
    if FC:
        if GRAY:
            model = torch.load("FC_gray_weights.pth", map_location=device)
        else:
            model = torch.load("FC_color_weights.pth", map_location=device)

        weights = {0: model["net.1.weight"],
                   1: model["net.3.weight"],
                   2: model["net.5.weight"]
                   }
        biases = {0: model["net.1.bias"],
                  1: model["net.3.bias"],
                  2: model["net.5.bias"]
                  }
        # print(weights)
        # print(weights[0][2][45])
    else:
        # read weights and biases from file
        model = torch.load("CNN_gray_weights.pth", map_location=device)

        weights = {0: model["net.0.weight"],
                   1: model["net.2.weight"],
                   2: model["net.5.weight"],
                   3: model["net.7.weight"],
                   4: model["net.10.weight"],
                   5: model["net.12.weight"],
                   6: model["net.15.weight"],
                   7: model["net.17.weight"],
                   8: model["net.21.weight"]
                   }
        biases = {0: model["net.0.bias"],
                  1: model["net.2.bias"],
                  2: model["net.5.bias"],
                  3: model["net.7.bias"],
                  4: model["net.10.bias"],
                  5: model["net.12.bias"],
                  6: model["net.15.bias"],
                  7: model["net.17.bias"],
                  8: model["net.21.bias"]
                  }

    utils.save_image(sample, f"original_{number}_{label}.png")
    sample = torch.unsqueeze(sample, dim=0)

    if FC:
        # find the adversarial instance
        adv = MILP.FC_MILP(weights, biases, sample, sample_label, GRAY)
    else:
        # find the adversarial instance
        adv = MILP_extreme.CNN_MILP(weights, biases, sample, sample_label)

    adv = torch.unsqueeze(adv, dim=0)

    utils.save_image(adv, f"adversarial_{number}_{label}.png")

    if GRAY:
        differences = torch.full((1, 1, 100, 100), 1.0, dtype=torch.float32) - torch.abs(sample - adv)
    else:
        differences = torch.full((1, 3, 100, 100), 1.0, dtype=torch.float32) - torch.abs(sample - adv)

    utils.save_image(differences, f"differences_{number}_{label}.png")

    images_pair = torch.cat((sample, adv, differences), dim=0)

    plt.figure()
    grid = utils.make_grid(images_pair)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.ioff()
    plt.show()

    utils.save_image(grid, f"comparison_{number}_{label}.png")


if __name__ == "__main__":
    main()
