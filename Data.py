from PIL import Image
from torch.utils.data import Dataset
import os


class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None, gray=False):
        super(MyDataset, self).__init__()
        images = []
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('/')[-2]) >= num_class:
                    break
                line = line.strip('\n')
                images.append(line)
                labels.append(int(line.split('/')[-2]))
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.gray = gray

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        if self.gray and image.mode != 'L':
            image = image.convert('L')
        elif not self.gray and image.mode == 'L':
            image = image.convert('RGB')
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)

def to_gray(image_path, save_path):
    image = Image.open(image_path).convert('L')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    image.save(save_path)
