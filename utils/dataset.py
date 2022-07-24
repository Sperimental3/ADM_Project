import os
import torch
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset


class CatsVsDogs(Dataset):
    """
    Cats vs. dogs dataset.
    """

    def __init__(self, root_dir, transform=None, target_transform=None, gray=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_transform (callable, optional): Optional transform to be applied
                only on the label of the sample.
            gray (bool): Telling if the images have to be taken colorized or not.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.gray = gray

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        """
        Take an item given an index.
        """

        img_name = os.path.join(self.root_dir, f"cat.{idx}.jpg" if idx <= 12499 else f"dog.{idx-12500}.jpg")

        if self.gray:
            image = io.imread(img_name, as_gray=True)
            image = np.expand_dims(image, -1)
        else:
            image = io.imread(img_name)

        if os.name == "nt":
            label = img_name.split("\\")[-1].split(".")[0]
        else:
            label = img_name.split("/")[-1].split(".")[0]

        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            label = self.target_transform(label)

        return {"image": sample["image"], "label": label}


class CatsVsDogsTest(Dataset):
    """
    Cats vs. dogs dataset for testing.
    """

    def __init__(self, root_dir, transform=None, gray=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            gray (bool): Telling if the images have to be taken colorized or not.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.gray = gray

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        """
        Take an item given an index.
        """

        img_name = os.path.join(self.root_dir, f"{idx+1}.jpg")

        if self.gray:
            image = io.imread(img_name, as_gray=True)
            image = np.expand_dims(image, -1)
        else:
            image = io.imread(img_name)

        # label here is inserted just to be compliant with the transformation methods that require both data and labels
        label = "placeholder"

        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample["image"]


class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample["image"]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = np.float32(transform.resize(image, (new_h, new_w)))

        return {"image": img, "label": sample["label"]}


class ToTensor(object):
    """
    Transform the numpy array into a normalized tensor.
    """

    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]

        image = torch.from_numpy(image.transpose((2, 0, 1)))

        return {"image": image, "label": label}


class EncodeLabel(object):
    """
    Convert class string label in a binary label.
    """

    def __call__(self, label):
        if label == "cat":
            binary_label = 0
        elif label == "dog":
            binary_label = 1
        else:
            print("Error with the label")

        return np.float32(binary_label)
