import torch
import nltk
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence
from nltk.translate.bleu_score import corpus_bleu
from model.visualization import Visualization


def evaluate(model, test_loader, vocab, device, criterion):  # TODO:add device
	model.eval()

	total_loss = 0.

	

	with torch.no_grad():
		for idx, batch in enumerate(iter(test_loader)):
			img, target = batch
			img = img.to(device)
			target = target.to(device)
			sentence_s = []
			target_s = []
			sentences = []
			attention_w = []
			for i in range(img.shape[0]):
				sentence, attention_w_t = model.inference(image=img[i].unsqueeze(0), vocab=vocab)
				sentence_s=sentence.split(' ')
				sentence_s.append('<END>')
				sentences.append(sentence_s)
				target_s.append(vocab.generate_caption(target[i,1:]).split(' '))
				attention_w.append(attention_w_t)
				
			total_loss += corpus_bleu(target_s,sentences,(1.0/1.0,))
			

			if idx % 10 == 0:
				example=' '.join(sentences[5])
				print(f'Evaluating batch {idx} / {len(test_loader)}...')
				print(f'Gen example: {example}')
				print(f'Exp example: {vocab.generate_caption(target[5,1:])}')
				Visualization.show_image(img[5],title=example)
				# Visualization.plot_attention(img[idx],sentences[idx],attention_w[idx])
				pass

		return total_loss / (idx+1)