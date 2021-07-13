import torch
import random
from panel.main import tensorboard_panel_eval
from nltk.translate.bleu_score import corpus_bleu
from model.visualization import Visualization
from einops.einops import rearrange

def write_on_tensorboard_evaluate(epoch:int, model, loss:int, expected_captions, generated_captions):
	convert_caption = lambda caption: model.vocab.generate_phrase(caption)
	expected_captions = list(map(convert_caption, expected_captions))
	generated_captions = list(map(convert_caption, generated_captions))

	tensorboard_panel_eval.add_sentences_comparison(epoch, expected_captions,generated_captions)
	#tensorboard_panel_eval.add_images(epoch, images, expected_captions, generated_captions)
	tensorboard_panel_eval.add_loss(epoch, loss)
	#tensorboard_panel.add_model_weights(epoch,model)
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
				
				sentences.append(sentence_s)
				target_s.append(vocab.generate_caption(target[i,1:]).split(' '))
				attention_w.append(attention_w_t)
				
			total_loss += corpus_bleu(target_s,sentences,(1.0/1.0,))
			
			

			if idx % 5 == 0:
				num_img=random.randint(0,img.shape[0]-1)
				example=' '.join(sentences[num_img])
				reference=vocab.generate_caption(target[num_img,1:])
				print(f'Evaluating batch {idx} / {len(test_loader)}...')
				print(f'Gen example: {example}')
				print(f'Exp example: {reference}')
			write_on_tensorboard_evaluate(epoch=idx+epoch,expected_captions=target[:,1:], generated_captions=sentences)

		return total_loss / (idx+1)

def evaluate_tr(model, test_loader, device, epoch,criterion):
	model.eval()

	total_loss = 0.
	
	with torch.no_grad():
		for idx, batch in enumerate(iter(test_loader)):
			img, target = batch
			img = img.to(device)
			target = target.to(device)

			

			aux=torch.ones(target.shape[0],1,dtype=int)*model.vocab.word_to_index['<PAD>']
			aux=aux.to(target.device)
			target=torch.cat([target,aux],dim=1)

			target_loss=target

			output = model(img, target[:,:-1])
			output = rearrange(
				output,
				'bsz seq_len vocab_size -> bsz vocab_size seq_len'
			)
			total_loss = criterion(output, target_loss[:,1:])
			

			if idx % 10 == 0:
				sentence = []
				num_img=random.randint(0,img.shape[0]-1)
				sentence = model.generate(image=img[num_img].unsqueeze(0))
				reference=model.vocab.generate_caption(target[num_img,1:])
				print(f'Evaluating batch {idx} / {len(test_loader)}...')
				print(f'Gen example (no teacher_forcing): {sentence}')
				print(f'Exp example: {reference}')
			#	string=str(num_img)+'_epoch_'+str(epoch)+'_plot.png'
			#	string_att=str(num_img)+'_epoch_'+str(epoch)+'_plot_att.png'
				#Visualization.show_image(img[num_img],title=example,fn=string)
			generated_captions = torch.argmax(output.transpose(1, 2), dim=-1)
			expected_captions = target[...,1:]
			generated_captions, expected_captions = generated_captions[:16,...], expected_captions[:16,...]
			
			write_on_tensorboard_evaluate(model= model,epoch=len(test_loader)*(epoch-1)+idx,loss=total_loss,expected_captions=expected_captions, generated_captions=generated_captions)
				
			#total_loss += corpus_bleu(target_s,sentences,(1.0/1.0,))			
				

		return total_loss

