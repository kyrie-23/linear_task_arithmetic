import os
import torch
import torchvision.datasets as datasets
import shutil
from pathlib import Path

class SUN397:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        traindir = os.path.join(location, 'sun397', 'train')
        valdir = os.path.join(location, 'sun397', 'val')


        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
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
        self.classnames = [idx_to_class[i][2:].replace('_', ' ') for i in range(len(idx_to_class))]


def process_dataset(txt_file, downloaded_data_path, output_folder):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        input_path = line.strip()
        final_folder_name = "_".join(x for x in input_path.split('/')[:-1])[1:]
        filename = input_path.split('/')[-1]
        output_class_folder = os.path.join(output_folder, final_folder_name)

        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        full_input_path = os.path.join(downloaded_data_path, input_path[1:])
        output_file_path = os.path.join(output_class_folder, filename)
        # print(final_folder_name, filename, output_class_folder, full_input_path, output_file_path)
        # exit()
        shutil.copy(full_input_path, output_file_path)
        if i % 100 == 0:
            print(f"Processed {i}/{len(lines)} images")
        

if __name__ == "__main__":
    downloaded_data_path = "~/data/sun397/SUN397"
    output_folder = "~/data/sun397"
    process_dataset(os.path.join(output_folder,'Training_01.txt'), downloaded_data_path, os.path.join(output_folder, "train"))
    process_dataset(os.path.join(output_folder,'Testing_01.txt'), downloaded_data_path, os.path.join(output_folder, "val"))