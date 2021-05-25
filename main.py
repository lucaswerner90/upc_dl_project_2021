import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset.main import Flickr8kDataset
from dataset.caps_collate import CapsCollate
from model.main import ImageCaptioningModel
from train import train
import json

CONFIGURATION = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()

with open(os.path.join(cwd, 'config.json')) as f:
    CONFIGURATION = json.load(f)

hparams = CONFIGURATION['HPARAMS']
hparams['device'] = device


def main():
    # Choosing image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(hparams['image_size']),
        transforms.CenterCrop(hparams['image_size']),
        # The normalize parameters depends on the model we're gonna use
        # If we apply transfer learning from a model that used ImageNet, then
        # we should use the ImageNet values to normalize the dataset
        # Otherwise we could just normalize the values between -1 and 1 using the
        # standard mean and standard deviation
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # Initializing & splitting dataset
    dataset = Flickr8kDataset(
        dataset_folder=CONFIGURATION['DATASET_FOLDER'],
        transform=transform,
        reduce=hparams['REDUCE_VOCAB'],
        vocab_max_size=hparams['VOCAB_SIZE']
    ).to(device)

    train_split, test_split = random_split(
        dataset, [32364, 8091])  # 80% train, 20% test

    # Initializing DataLoaders
    train_loader = DataLoader(
        train_split,
        batch_size=hparams['BATCH_SIZE'],
        collate_fn=CapsCollate(
            pad_idx=dataset.vocab.word_to_index['<PAD>'], batch_first=True)
    ).to(device)
    test_loader = DataLoader(
        test_split,
        batch_size=hparams['BATCH_SIZE'],
        collate_fn=CapsCollate(
            pad_idx=dataset.vocab.word_to_index['<PAD>'], batch_first=True)
    ).to(device)

    # instantiating model
    model = ImageCaptioningModel(
        embed_size=hparams['EMBED_SIZE'],
        vocab_size=len(dataset.vocab.word_to_index),
        caption_max_length=hparams['MAX_LENGTH']
    ).to(device)

    # select optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=hparams['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss(
        ignore_index=dataset.vocab.word_to_index['<PAD>'])

    # LET'S TRAIN!!!
    train(
        num_epochs=hparams['NUM_EPOCHS'],
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=hparams['device'],
        log_interval=['log_interval']
    )


if __name__ == "__main__":

    image_size = (128, 128)
    batch_size = 2
    MAX_SIZE = 5000
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(hparams['IMAGE_SIZE']),
        transforms.RandomCrop(hparams['IMAGE_SIZE']),
        # The normalize parameters depends on the model we're gonna use
        # If we apply transfer learning from a model that used ImageNet, then
        # we should use the ImageNet values to normalize the dataset.
        # Otherwise we could just normalize the values between -1 and 1 using the
        # standard mean and standard deviation
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = Flickr8kDataset(dataset_folder=CONFIGURATION['DATASET_FOLDER'], transform=transform,
                              reduce=hparams['REDUCE_VOCAB'], vocab_max_size=hparams['VOCAB_SIZE'])

    # Test the dataloader
    model = ImageCaptioningModel(
        image_features_dim=hparams['IMAGE_FEATURES_DIM'],
		embed_size=hparams['EMBED_SIZE'],
        vocab_size=len(dataset.vocab.word_to_index),
        caption_max_length=hparams['MAX_LENGTH'],
		device = hparams['DEVICE']
    ).to(device)

    train_split, test_split = random_split(
        dataset, [32364, 8091])  # 80% train, 20% test
    train_loader = DataLoader(train_split, batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(
        pad_idx=dataset.vocab.word_to_index['<PAD>'], batch_first=True))
    test_loader = DataLoader(test_split, batch_size=hparams['BATCH_SIZE'], collate_fn=CapsCollate(
        pad_idx=dataset.vocab.word_to_index['<PAD>'], batch_first=True))

    optimizer = optim.Adam(model.parameters(), lr=hparams['LEARNING_RATE'])
    criterion = nn.CrossEntropyLoss(
        ignore_index=dataset.vocab.word_to_index['<PAD>'])

    train(
        num_epochs=hparams['NUM_EPOCHS'],
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=hparams['device'],
        log_interval=hparams['LOG_INTERVAL'],
        vocab=dataset.vocab
    )
