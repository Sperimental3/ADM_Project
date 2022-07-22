from skimage import io
import torch
from torchvision.transforms import ToTensor
from utils import NN


def main():

    FC = False
    GRAY = True

    original = io.imread("original_746_gray.png", as_gray=GRAY)
    transform = ToTensor()
    original = transform(original).type(torch.float32)
    original = torch.unsqueeze(original, 0)

    print(original.shape, original.dtype, original)

    adversarial = io.imread("prova.png", as_gray=GRAY)
    adversarial = transform(adversarial).type(torch.float32)
    adversarial = torch.unsqueeze(adversarial, 0)

    if FC:
        if GRAY:
            # define net
            net = NN.Weak(1 * 100 * 100, 1)

            # load weights and biases of the previously trained net
            net.load_state_dict(torch.load("FC_gray_weights.pth"))
            net = net.eval()
        else:
            # define net
            net = NN.Weak(3 * 100 * 100, 1)

            # load weights and biases of the previously trained net
            net.load_state_dict(torch.load("FC_color_weights.pth"))
            net = net.eval()
    else:
        # define net
        if GRAY:
            net = NN.Strong(1, 1)

            # load weights and biases of the previously trained net
            net.load_state_dict(torch.load("CNN_gray_weights.pth"))
            net = net.eval()
        else:
            net = NN.Strong(3, 1)

            # load weights and biases of the previously trained net
            net.load_state_dict(torch.load("CNN_color_weights.pth"))
            net = net.eval()

    pred = net(original)
    print(pred)
    org_pred = pred.round()
    if org_pred == 1:
        org_label = "dog"
    else:
        org_label = "cat"

    pred = net(adversarial)
    print(pred)
    adv_pred = pred.round()
    if adv_pred == 1:
        adv_label = "dog"
    else:
        adv_label = "cat"

    print(f"The predicted label for the original instance is: {org_label},"
          f" the predicted one for the adversarial instance is: {adv_label}")


if __name__ == "__main__":
    main()
