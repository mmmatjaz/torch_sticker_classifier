import glob
import os
import random
from enum import Enum, IntEnum

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image
from google.cloud import storage
from pathlib import Path

from torchvision.transforms import transforms


class Label(IntEnum):
    BOX, DEVICE = range(2)

    def prefix(self):
        return ["qr-box", "qr-device"][self.value]


GCP_KEY_PATH = Path.home().joinpath(".gripable-dev-pipes-keys").joinpath("gripable-calib-key.json")
IMAGE_SIZE = (96, 54)


class SubSet(Enum):
    TRAIN, TEST, VALID = range(3)


class QrData(Dataset):
    def __init__(self, subset: SubSet, cache_dir=Path.home().joinpath(".qr_dataset"), transform=None,
                 target_transform=None, force_reload=True):

        # build list of ignore data
        ignore = []
        for txt in glob.glob("*.txt"):
            with open(txt, 'r') as ftxt:
                ignore += [l.strip() for l in ftxt.readlines()]

        st_client = storage.Client.from_service_account_json(str(GCP_KEY_PATH))
        self.bucket = bucket = st_client.get_bucket('gripable-calib.appspot.com')

        self.img_dir = cache_dir
        df_cache_path = self.img_dir.joinpath("qr_data.csv")
        if force_reload or not os.path.exists(df_cache_path):
            if not os.path.exists(df_cache_path.parent):
                os.makedirs(df_cache_path.parent)

            blobs = []
            for lbl in Label:
                for b in bucket.list_blobs(prefix=lbl.prefix()):
                    bb = b.name.split("/")[1]
                    if bb not in ignore:
                        blobs.append((b.name, lbl))
                        #print(bb)
            """
            for b in bucket.list_blobs(prefix="qr-device"):
                bb = b.name.split("/")[1]
                if bb in ignore:
                    blobs.append((b.name, Label.DEVICE))
            """
            self.blobs = pd.DataFrame(blobs, columns=["path", "label"])
            self.blobs.to_csv(df_cache_path)
        else:
            self.blobs = pd.read_csv(df_cache_path, index_col=0)

        probs = np.random.rand(len(self.blobs))
        training_mask = probs < 0.7
        test_mask = (probs >= 0.7) & (probs < 0.85)
        validatoin_mask = probs >= 0.85

        if subset == SubSet.TRAIN:
            self.blobs = self.blobs[training_mask]
        elif subset == SubSet.TEST:
            self.blobs = self.blobs[test_mask]
        else:
            self.blobs = self.blobs[validatoin_mask]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.blobs)

    def __getitem__(self, idx):
        blob_name, label = self.blobs.iloc[idx]
        cache_path = self.img_dir.joinpath(f"{Label(label).prefix()}", blob_name.split("/")[1])
        # print(blob_name, label, cache_path)
        if not os.path.exists(cache_path):
            if not os.path.exists(cache_path.parent):
                os.makedirs(cache_path.parent)
            self.bucket.blob(blob_name).download_to_filename(cache_path)
            print(f"downloading {blob_name}")
        image = read_image(str(cache_path))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def show_data(data_sample):
    plt.imshow(np.swapaxes(data_sample[0].numpy(), 0, 2))
    plt.title('y = ' + Label(data_sample[1]).name)


composed_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(IMAGE_SIZE),
                                         transforms.ToTensor(),
                                         ])  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if __name__ == "__main__":
    data = QrData(SubSet.TRAIN, transform=composed_transform)
    img, label = sample = data[random.randint(0, len(data))]
    show_data(sample)
