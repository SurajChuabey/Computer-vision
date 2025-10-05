from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.datapreprocessing.preprocessing import CFSIRDataset
from src.constant.constant import Constants

class CFSIRDataLoader:
    @staticmethod
    def load_data(data_dir,batch_size = 8):

        train_dataset = CFSIRDataset(data_dir, train=True,transform=CFSIRDataLoader.transform(train=True))
        test_dataset = CFSIRDataset(data_dir, train=False,transform=CFSIRDataLoader.transform(train=False))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for images, labels in train_loader:
            print(images.shape, labels.shape)
            break
        return train_loader,test_loader

    @staticmethod
    def transform(train = True):
        if train:
            transform = transforms.Compose([
                transforms.ToPILImage(),                     # Convert tensor to PIL image
                transforms.RandomHorizontalFlip(),           # Random horizontal flip
                transforms.RandomCrop(32, padding=4),       # Random crop with padding
                transforms.ToTensor(),                       # Convert back to tensor
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],         # CIFAR mean
                    std=[0.2470, 0.2435, 0.2616]           # CIFAR std
                )
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                )
            ])
        return transform