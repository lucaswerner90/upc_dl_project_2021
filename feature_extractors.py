## In this file few feature extraction functions are defined. Such functions don't have input arguments
#  and return a Composition of transforms to be applied to the database as well as a feature extraction module.
#  The feature extraction module, given as input a tensor with a badge of cropped and normalized RGB images, returns a batch of tensors of dimesion (bsz,num_feat,L) where L are the number 
#  of embeddings of the image (each one roughly corresponding to a patch of the image) 

## Ojo, que mejor se debería hacer uns tranpose de los tensores de salida. Eso segun se haya definido el attention y el decoder.


import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

def get_feat_extractor_Vgg_L49():
    """
    This function returns a Composition of transforms to be applied to the images of the database, and 
    a feature extraction module which is the convolutional base of a VGG-16 Neural Network and can be used as encoder.
    The feature extraction module needs at its input a tensor with a badge of 224x224 cropped RGB images normalized with the Imagenet 
    parameters. Input tensor dimensions (bsz,3,224,224). And gives as output a tensor with a badge of L vectors of features (embeddings), each vector 
    roughly corresponding to the embedding of a patch of the image. Dimesion: (bsz,num_feat=512,L=49)
    The feature extraction model has all its layers frozen.
    Example:
    transf, encoder = get_feat_extractor_Vgg_L49()
	encoder.eval()      
    encoder.to(device)
    """

## Aquests funció retorna la tranformació que s'ha d'aplicar a la base de dades i un modul amb la part convolucuinal de una pretrained_VGG més un flatten per aconseguir 49 vectores (7x7) de 512 feaures cadascun
# El model té les capes congelades ja que està pensat inicialment per a inferència. 
# La transformació conté una normalització amb valors de Imagenet ja que la pretrained VGG estava entrenada amb Imagenet.

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
                           transforms.Resize(224),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           normalize,
    ])

    pretrained_model = models.vgg16(pretrained=True)

    feat_extractor = nn.Sequential(pretrained_model.features,nn.Flatten(2,3))


    for layer in feat_extractor[:]:  # Freeze All layers 
        for param in layer.parameters():
            param.requires_grad = False

#for layer in feature_extractor[24:]:  # Train layers 24 to 30
#    for param in layer.parameters():
#        param.requires_grad = True
    return trans, feat_extractor




#feature_extractor = nn.Sequential(modelAux[0], modelAux[1], modelAux[2])
#feature_extractor = nn.Sequential(modelAux[0:31])
#feature_extractor = nn.Sequential(*list(modelAux)[0:23])

def get_feat_extractor_Vgg_L196():

    """
    This function returns a Composition of transforms to be applied to the images of the database, and 
    a feature extraction module which is the convolutional base of a VGG-16 Neural Network trunked a given layer, and can be used as encoder.
    The feature extraction module needs at its input a tensor with a badge of 224x224 cropped RGB images normalized with the Imagenet 
    parameters. Input tensor dimensions (bsz,3,224,224). And gives as output a tensor with a badge of L vectors of features (embeddings), each vector 
    roughly corresponding to the embedding of a patch of the image. Dimesion: (bsz,num_feat=512,L=196)
    The feature extraction model has all its layers frozen.
    Example:
    transf, encoder = get_feat_extractor_Vgg_L196()
    """
## Aquests funció retorna la tranformació que s'ha d'aplicar a la base de dades i un modul amb la part convolucuinal de una pretrained_VGG més un flatten per aconseguir 49 vectores (7x7) de 512 feaures cadascun
# El model té les capes congelades ja que està pensat inicialment per a inferència. 
# La transformació conté una normalització amb valors de Imagenet ja que la pretrained VGG estava entrenada amb Imagenet.

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
                           transforms.Resize(224),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           normalize,
    ])

    pretrained_model = models.vgg16(pretrained=True)

    #feature_extractor = nn.Sequential(modelAux[0], modelAux[1], modelAux[2])
    modelAux = pretrained_model.features
    modelAux = nn.Sequential(*list(modelAux)[0:24])

    feat_extractor = nn.Sequential(modelAux,nn.Flatten(2,3))


    for layer in feat_extractor[:]:  # Freeze All layers 
        for param in layer.parameters():
            param.requires_grad = False

#for layer in feature_extractor[24:]:  # Train layers 24 to 30
#    for param in layer.parameters():
#        param.requires_grad = True
    return trans, feat_extractor

