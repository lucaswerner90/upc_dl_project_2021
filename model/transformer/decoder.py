from einops import rearrange
import torch
import math
from torch.functional import Tensor

from torch import nn
from model.transformer.positional_encoding import PositionalEncoding

class TransformerDecoder(nn.Module):
    def __init__(self,image_features_dim,vocab_size, embed_size, num_layers=4):
        super(TransformerDecoder,self).__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.image_features_dim = image_features_dim
        self.embedding_size = embed_size
        self.embed = nn.Embedding(vocab_size, self.embedding_size)
        self.positional_encoder = PositionalEncoding(emb_size=self.embedding_size)
        

        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model = self.embedding_size,
            dim_feedforward=512,
            activation='relu',
            dropout=.5,
            nhead=4
        )

        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer,
            num_layers=self.num_layers
        )

        self.reduce_features = nn.Linear(in_features=self.image_features_dim, out_features = self.embedding_size)

        self.linear = nn.Linear(in_features= self.embedding_size, out_features=self.vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, image_features: Tensor, captions: Tensor):

        embeddings = self.embed(captions) * math.sqrt(self.embedding_size)
        
        image_features = self.reduce_features(image_features.flatten(1,2))
        embeddings = rearrange(
            embeddings,
            'bsz seq_len embed_size -> seq_len bsz embed_size'
        )
        positional_embeddings = self.positional_encoder(embeddings)
        
        outputs = self.transformer_decoder.forward(
            tgt=positional_embeddings,
            memory=image_features,
            tgt_mask=self.generate_square_subsequent_mask(len(positional_embeddings))
        )
        outputs = rearrange(
            outputs, 
            'seq_len bsz embed_size -> bsz seq_len embed_size'
        )
        y = self.linear(outputs)
        return y