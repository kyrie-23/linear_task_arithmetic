import os
import torch
import torchvision.datasets as datasets
import re
import shutil
import random

def pretify_classname(classname):
    l = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', classname)
    l = [i.lower() for i in l]
    out = ' '.join(l)
    if out.endswith('al'):
        return out + ' area'
    return out

class EuroSATBase:
    def __init__(self,
                 preprocess,
                 test_split,
                 location='~/data',
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        traindir = os.path.join(location, 'EuroSAT_splits', 'train')
        testdir = os.path.join(location, 'EuroSAT_splits', test_split)


        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(testdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
        self.classnames = [pretify_classname(c) for c in self.classnames]
        ours_to_open_ai = {
            'annual crop': 'annual crop land',
            'forest': 'forest',
            'herbaceous vegetation': 'brushland or shrubland',
            'highway': 'highway or road',
            'industrial area': 'industrial buildings or commercial buildings',
            'pasture': 'pasture land',
            'permanent crop': 'permanent crop land',
            'residential area': 'residential buildings or homes or apartments',
            'river': 'river',
            'sea lake': 'lake or sea',
        }
        for i in range(len(self.classnames)):
            self.classnames[i] = ours_to_open_ai[self.classnames[i]]


class EuroSAT(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='~/data',
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, 'test', location, batch_size, num_workers)


class EuroSATVal(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='~/data',
                 batch_size=32,
                 num_workers=16):
        super().__init__(preprocess, 'val', location, batch_size, num_workers)


def create_directory_structure(base_dir, classes):
    for dataset in ['train', 'val', 'test']:
        path = os.path.join(base_dir, dataset)
        os.makedirs(path, exist_ok=True)
        for cls in classes:
            os.makedirs(os.path.join(path, cls), exist_ok=True)

def split_dataset(base_dir, source_dir, classes, val_size=270, test_size=270):
    for cls in classes:
        class_path = os.path.join(source_dir, cls)
        images = os.listdir(class_path)
        random.shuffle(images)

        val_images = images[:val_size]
        test_images = images[val_size:val_size + test_size]
        train_images = images[val_size + test_size:]

        for img in train_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, 'train', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
        for img in val_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, 'val', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
        for img in test_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(base_dir, 'test', cls, img)
            print(src_path, dst_path)
            shutil.copy(src_path, dst_path)
            
if __name__ == "__main__":           
    source_dir = '~/data/eurosat/2750'  # replace with the path to your dataset
    base_dir = '~/data/EuroSAT_Splitted'  # replace with the path to the output directory

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    create_directory_structure(base_dir, classes)
    split_dataset(base_dir, source_dir, classes)