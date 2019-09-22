import sys
import os
print(os.listdir('./satelliate-enhancer/'))
sys.path.append('./satelliate-enhancer/')

import argparse
import os
from satellite_enhancer.trainer import Trainer
from satellite_enhancer.dataset.satellite_dataset import SatelliteDataset
from satellite_enhancer.model.generator import Generator
from satellite_enhancer.model.discriminator import Discriminator


def main():

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    parser = argparse.ArgumentParser(description='Train Satellite super resolution')
    parser.add_argument('--dataset-path', dest="dataset_path", type=dir_path,
                        help='Path to satellite dataset', required=True)

    arguments = parser.parse_args()
    print(arguments.dataset_path)

    satellite_train = SatelliteDataset(scale=4, images_dir=arguments.dataset_path)
    train_ds = satellite_train.dataset(batch_size=16, random_transform=True)

    trainer = Trainer(Generator(), Discriminator())

    trainer.train(train_ds, num_epochs=500)


if __name__ == '__main__':
    main()

