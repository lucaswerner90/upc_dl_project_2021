import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
	def __init__(self, image_features_dim, decoder_hidden_state_dim, attention_dim):
		super(Attention, self).__init__()

		self.attention_dim = attention_dim

		self.U = nn.Linear(in_features=image_features_dim, out_features=attention_dim)
		self.W = nn.Linear(in_features=decoder_hidden_state_dim,out_features=attention_dim)

		self.A = nn.Linear(attention_dim,1)
	
	def forward(self, features, hidden_state):
		u_features = self.U(features.transpose(1,2))
		w_hidden_state = self.W(hidden_state)

		combined_states = torch.tanh(u_features + w_hidden_state.transpose(0,1))
		attention_scores = self.A(combined_states)
		attention_scores = attention_scores.squeeze(2)

		alphas = F.softmax(attention_scores, dim=1) 
		weighted_features = features.transpose(1,2) * alphas.unsqueeze(2)
		weighted_features = weighted_features.sum(dim=1)

		return alphas, weighted_features
