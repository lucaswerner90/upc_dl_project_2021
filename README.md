
# Image Captioning - UPC AI w/ DL 2021
![upc logo](https://fiorp.org/wp-content/uploads/2016/11/logo-cc-upc.jpg)

## Table of contents
- [Image Captioning - UPC AI w/ DL 2021](#image-captioning---upc-ai-w-dl-2021)
	- [Table of contents](#table-of-contents)
	- [Description](#description)
	- [Installation](#installation)
		- [Create the environment:](#create-the-environment)
		- [Install the dependencies:](#install-the-dependencies)
	- [Download the dataset](#download-the-dataset)
	- [Train the model](#train-the-model)
	- [Authors](#authors)
## Description

## Installation

### Create the environment:
```
python3 -m venv .env
source .env/bin/activate
```
### Install the dependencies:
```
pip install --no-chache-dir -r requirements.txt
```

## Download the dataset

The dataset used during the training phase is the [Flickr8k](https://www.kaggle.com/adityajn105/flickr8k/).
It consists of 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations.

```
python dataset/download.py
```

## Train the model

```
python main.py
```

## Authors
* Lucas Werner
* Adriá Molero
* Pere Pujol
* Rai Gordejuela