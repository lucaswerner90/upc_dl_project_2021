import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms

from dataset.vocabulary import Vocabulary
from dataset.caps_collate import CapsCollate

class Flickr8kDataset(Dataset):
    """
    Dataset for Flickr8k data treatment
    """

    def __init__(self, dataset_folder,transform=None,reduce=False,vocab_max_size=5000,feature_extractor=None):
        super(Flickr8kDataset, self).__init__()
        # assert (feature_extractor!=None) and (transform is None), "Both Feature_extractor and transform cannot be different than None"
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.images_folder = os.path.join(dataset_folder,'Images')
        self.dataframe = pd.read_csv(open(os.path.join(dataset_folder,'captions.txt'),'r'))

        self.caption = self.dataframe['caption']
        self.vocab = Vocabulary()
        self.vocab.build_vocabulary(self.caption.tolist(),reduce,vocab_max_size)
        

    def read_image(self,filename:str):
        """
        Reads the image file based on the filename

        Returns:
            [Image]: Returns a PIL.Image object 
        """
        image_path = os.path.join(os.getcwd(),self.images_folder, filename)
        image = Image.open(image_path)
        return image

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,idx):
        """
        Process the dataset according to our needs.
        The image needs to be resize and normalize.

        Args:
            idx (int): Index of the dataset element to be returned

        Returns:
            torch.Tensor: 	Processed image
            string:			Caption of the image
        """
        filename, caption = self.dataframe.iloc[idx,:]
        image = self.read_image(filename)

        if self.transform:
            image = self.transform(image)
        elif self.feature_extractor:
            image = self.feature_extractor(images=image, return_tensors="pt").data['pixel_values'].squeeze()

        tok_vec = []
        tok_vec += [self.vocab.word_to_index['<START>']]
        tok_vec += self.vocab.numericalize(caption)
        tok_vec += [self.vocab.word_to_index['<END>']]

        return image, torch.tensor(tok_vec)


if __name__ == "__main__":
    image_size = (128,128)
    batch_size = 128
    MAX_SIZE = 5000
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        # The normalize parameters depends on the model we're gonna use
        # If we apply transfer learning from a model that used ImageNet, then
        # we should use the ImageNet values to normalize the dataset.
        # Otherwise we could just normalize the values between -1 and 1 using the 
        # standard mean and standard deviation
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = Flickr8kDataset(dataset_folder='data',transform=transform,reduce=True,vocab_max_size=MAX_SIZE)

    # Test the dataloader
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size, collate_fn=CapsCollate(pad_idx=dataset.vocab.word_to_index['<PAD>'],batch_first=True))
    batch_sample = next(iter(dataloader))
    images, captions = batch_sample

    assert images.size() == (batch_size,) + (3,) + image_size
    assert len(captions) == batch_size

    for image in images:
        assert image.shape == (3,) + image_size 
