import torch
import random
from nltk.translate.bleu_score import corpus_bleu
from model.visualization import Visualization


def evaluate(model, test_loader, vocab, device, epoch):
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
				sentence, attention_w_t = model.inference(image=img[i].unsqueeze(0))
				sentence_s=sentence.split(' ')
				sentence_s.append('<END>')
				sentences.append(sentence_s)
				target_s.append(vocab.generate_caption(target[i,1:]).split(' '))
				attention_w.append(attention_w_t)
				
			total_loss += corpus_bleu(target_s,sentences,(1.0/1.0,))
			

			if idx % 10 == 0:
				num_img=random.randint(0,img.shape[0]-1)
				example=' '.join(sentences[num_img])
				reference=vocab.generate_caption(target[num_img,1:])
				print(f'Evaluating batch {idx} / {len(test_loader)}...')
				print(f'Gen example: {example}')
				print(f'Exp example: {reference}')
				string=str(num_img)+'_epoch_'+str(epoch)+'_plot.png'
				string_att=str(num_img)+'_epoch_'+str(epoch)+'_plot_att.png'
				Visualization.show_image(img[num_img],title=example,fn=string)
				Visualization.plot_attention(img[num_img],sentences[num_img][:-1],attention_w[num_img],fn=string_att)
				
				

		return total_loss / (idx+1)
