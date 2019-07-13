import glob

import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class AlignedDataset(data.Dataset):
    def __init__(self, data_dir, direction, img_size):
        super(AlignedDataset, self).__init__()

        self.AB_dir = '{}'.format(data_dir)
        self.AB_paths = glob.glob('{}/*'.format(self.AB_dir))
        self.direction = direction

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.AB_paths)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        w, h = AB.size
        w2 = int(w / 2)

        if self.direction == 'AtoB':
            A = AB.crop((0, 0, w2, h))
            B = AB.crop((w2, 0, w, h))
        else:
            A = AB.crop((w2, 0, w, h))
            B = AB.crop((0, 0, w2, h))

        A = self.transform(A)
        B = self.transform(B)

        return A, B
