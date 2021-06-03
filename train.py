import torch
import os
from model.visualization import Visualization
from panel.main import tensorboard_panel


def write_on_tensorboard(epoch:int, loss:int, bleu:int, image, expected_captions, generated_captions):
	tensorboard_panel.add_sentences_comparison(epoch,expected_captions[0],generated_captions[0])
	tensorboard_panel.add_loss(epoch,loss)
	tensorboard_panel.add_bleu(epoch,bleu)
	tensorboard_panel.add_image(epoch,image,expected_captions[0],generated_captions[0])


def train_single_epoch(epoch, model, train_loader, optimizer, criterion, device):
	"""
	Train single epoch
	"""
	model.train()
	for i, batch in enumerate(iter(train_loader)):
		img, target = batch
		img, target = img.to(device), target.to(device)

		optimizer.zero_grad()

		output, _ = model(img, target)
		loss = criterion(output, target[:,1:])
		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)

		optimizer.step()

		candidate_corpus = [model.vocab.generate_caption(torch.argmax(output[0].transpose(1, 0), dim=-1))]
		reference_corpus = [model.vocab.generate_caption(target[0, 1:])]
		bleu = 0
		# bleu = bleu_score(candidate_corpus, reference_corpus)
		write_on_tensorboard(i+(epoch*len(train_loader)),loss.item(),bleu,img[0],reference_corpus,candidate_corpus)

def evaluate(model,test_loader, vocab, device,criterion):
	model.eval()

	total_loss = 0.
	#device= 'cpu'
	with torch.no_grad():
		for idx, batch in enumerate(iter(test_loader)):
			img, target = batch
			img = img.to(device)
			target = target.to(device)
			for i in range(img.shape[0]):
				sentence = model.inference(image=img[i].unsqueeze(0),vocab=vocab)
				alphas = model.forward(image=img[i].unsqueeze(0), vocab=vocab)[1]
			

			caption = ' '.join(sentence)
			Visualization.plot_attention((img[0]), sentence, alphas) # showing expected and plotting attention
			total_loss += target.numel()*criterion(sentence,target).item()
			n += target.numel()

		return total_loss / n, caption


def save_model(model, epoch):
	"""
	Function to save current model
	"""
	filename = os.path.join('model','checkpoints','Epoch_'+str(epoch)+'_model_state.pth')
	model_state = {
		'epoch':epoch,
		'model':model.state_dict()
	}
	torch.save(model_state, filename)

def train(num_epochs, model, train_loader,test_loader, optimizer, criterion, device,log_interval,vocab):
	"""
	Executes model training. Saves model to a file every 5 epoch.
	"""	

	for epoch in range(1,num_epochs+1):
		train_single_epoch(epoch, model, train_loader,optimizer, criterion, device)

		if epoch % 5 == 0:
			save_model(model, epoch)
	
