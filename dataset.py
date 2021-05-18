import os
import nltk
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from torchvision import transforms


class Vocabulary():
    def __init__(self):
        #initialize known tokens
        self.index_to_word = {0: "<PAD>", 1:"<START>", 2:"<END>",3:"<UNK>"}

        self.word_to_index = {v:k for k, v in self.index_to_word.items()}

    def __len__(self):
        return len(self.word_to_index)

    @staticmethod
    def tokenize(text):
        return nltk.word_tokenize(text)
    
    def build_vocabulary(self,sentence_list,reduce=True, max_size=5000):

        idx = 4 #starting idx
        nltk.download("punkt")
        
        wtoi = {}

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                wtoi[word] = idx
                idx += 1
        if reduce:
            items=list(wtoi.items())
            items.sort(key=lambda x:x[1],reverse=True)
            items=items[:max_size]
            itow={}
            idx=4
            for k,v in items:
                itow[idx]=k
                idx+=1
            #wtoi = {k:v for k,v in items}
            self.index_to_word={**self.index_to_word,**itow}
        else:
            self.index_to_word={**self.index_to_word,**wtoi}
        self.word_to_index = {v:k for k, v in self.index_to_word.items()}

    def numericalize(self,text):
        tok_text = self.tokenize(text)
        return [self.word_to_index[token] if token in self.word_to_index else self.word_to_index['<UNK>'] for token in tok_text]

 
class Flickr8kDataset(Dataset):

    def __init__(self, dataset_folder='/local/Datasets/8KFlicker/archive',transform=None):
        super().__init__()
        self.transform = transform
        self.images_folder = os.path.join(dataset_folder,'Images')
        self.dataframe = pd.read_csv(open(os.path.join(dataset_folder,'captions.txt'),'r'))

        self.caption = self.dataframe['caption']
        self.vocab = Vocabulary()
        self.vocab.build_vocabulary(self.caption.tolist(),reduce=True,max_size=5000)
        

    def read_image(self,filename):
        """
        Reads the image file based on the filename

        Args:
            filename (string): Filename that contains the image

        Returns:
            [Image]: Returns a PIL.Image object 
        """
        image_path = os.path.join(self.images_folder, filename)
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

        tok_vec = []
        tok_vec += [self.vocab.word_to_index['<START>']]
        tok_vec += self.vocab.numericalize(caption)
        tok_vec += [self.vocab.word_to_index['<END>']]

        return image, torch.tensor(tok_vec)
class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs,targets

if __name__ == "__main__":
    image_size = (128,128)
    batch_size = 128
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
    dataset = Flickr8kDataset(transform=transform)

    # Test the dataloader
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size, collate_fn=CapsCollate(pad_idx=dataset.vocab.word_to_index['<PAD>'],batch_first=True))
    batch_sample = next(iter(dataloader))
    images, captions = batch_sample

    assert images.size() == (batch_size,) + (3,) + image_size
    assert len(captions) == batch_size

    for image in images:
        assert image.shape == (3,) + image_size 
