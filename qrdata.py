import glob
import os
import random
from enum import Enum, IntEnum

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from google.cloud import storage
from pathlib import Path

from torchvision.transforms import transforms

from crop_to_qr import crop_image
from qr_torch.helpers import show_data


class Label(IntEnum):
    BOX, DEVICE = range(2)

    def prefix(self):
        return ["qr-box", "qr-device"][self.value]


GCP_KEY_PATH = Path.home().joinpath(".gripable-dev-pipes-keys").joinpath("gripable-calib-key.json")
# IMAGE_SIZE = (96, 54)
IMAGE_SIZE = (64, 64)


class SubSet(Enum):
    TRAIN, TEST, VALID = range(3)


class QrData(Dataset):
    probs = None

    def __init__(self, subset: SubSet, cache_dir=Path.home().joinpath(".qr_dataset"), transform=None,
                 target_transform=None, force_reload=True, auto_crop=True):
        self.crop=auto_crop
        # build list of ignore data
        ignore = []
        for txt in glob.glob("data_ignore/*.txt"):
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
                        # print(bb)
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

        # only split the dataset (into train, test, valid subsets) once
        if self.probs is None or len(self.blobs) != len(self.probs):
            self.probs = np.random.rand(len(self.blobs))

        if subset == SubSet.TRAIN:
            training_mask = self.probs < 0.7
            self.blobs = self.blobs[training_mask]
        elif subset == SubSet.TEST:
            test_mask = (self.probs >= 0.7) & (self.probs < 0.85)
            self.blobs = self.blobs[test_mask]
        else:
            validatoin_mask = self.probs >= 0.85
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

        image_tensor = read_image(str(cache_path))
        if self.crop:
            a, b, c, d = crop_image(str(cache_path))
            image_tensor = image_tensor[:, a:b, c:d]

        try:
            if self.transform:
                image_tensor = self.transform(image_tensor)
        except ValueError as e:
            print(cache_path)
            raise e
        if self.target_transform:
            label = self.target_transform(label)
        return image_tensor, label


composed_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(IMAGE_SIZE),
                                         transforms.ToTensor(),
                                         ])  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if __name__ == "__main__":
    ct = transforms.Compose([transforms.ToPILImage(),
                             transforms.ToTensor(),
                             ])  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = QrData(SubSet.TRAIN, transform=ct)
    ind = random.randint(0, len(data))
    img, label = sample = data[ind]
    show_data(sample)
