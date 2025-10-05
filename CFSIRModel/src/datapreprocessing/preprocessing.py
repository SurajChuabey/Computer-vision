from torch.utils.data import Dataset
from src.constant.constant import Constants
import pickle
import numpy as np
import torch
import os

class CFSIRDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        files = [f"{Constants.DATA_BATCH}{i}" for i in range(1, 6)] if train else [Constants.TEST_BATCH]

        # Load all batches
        for file in files:
            file_path = os.path.join(data_dir, file)
            with open(file_path, Constants.FILE_OPENING_FORMAT) as f:
                batch = pickle.load(f, encoding=Constants.ENCODING)
                images = batch[b'data']
                labels = batch[b'labels']

                # Reshape to (N, 3, 32, 32)
                images = images.reshape(len(images), 3, 32, 32)

                self.data.append(images)
                self.labels.extend(labels)

        # Concatenate all data batches
        self.data = np.concatenate(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":
    dataset = CFSIRDataset(Constants.MODEL_PATH)
    print(f"Dataset length: {len(dataset)}")
    sample_img, sample_label = dataset[0]
    print(f"Sample image shape: {sample_img.shape}, Sample label: {sample_label}")
