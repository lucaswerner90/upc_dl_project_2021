import time
import torch
import os
from model.visualization import Visualization
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

writer = SummaryWriter()

def write_on_tensorboard(writer:SummaryWriter, epoch:int, loss:int, bleu:int, image, expected_captions, generated_captions):
	writer.add_text('Expected', expected_captions[0], epoch)
	writer.add_text('Generated', generated_captions[0], epoch)
	writer.add_scalar('Loss/train', loss, epoch)

	image = Visualization.process_image(image)
	plt.tight_layout()
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.set_title(f'Exp: {expected_captions[0]}\n Gen: {generated_captions[0]}',fontsize=8)
	ax.imshow(image)
	writer.add_figure('Generated captions',fig, global_step=epoch)


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
		write_on_tensorboard(writer,i*(epoch+1),loss.item(),bleu,img[0],reference_corpus,candidate_corpus)

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
	
