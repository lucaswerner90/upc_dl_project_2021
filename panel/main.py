from typing import List
from math import sqrt
from textwrap import wrap
from torch.functional import Tensor
from torch.utils.tensorboard import SummaryWriter
from model.visualization import Visualization

import matplotlib.pyplot as plt

class Tensorboard:
	def __init__(self,directory:str='runs/ImageCaptioning') -> None:
		self.writer = SummaryWriter(directory)

	def add_sentences_comparison(self,epoch:int, expected_captions:List[str], generated_captions:List[str]):
		text = '| Expected | Generated | \n |--- |--- | \n'
		for i in range(len(expected_captions)):
			# text += f'**Expected:**\n\n {} \n\n **Generated**:\n\n{generated_captions[i]} \n\n'
			text += f'| {expected_captions[i]} | {generated_captions[i]} |\n'

		self.writer.add_text('Expected/Generated', text, epoch)

	def add_model_weights(self,epoch, model):
		for name,parameters in model.decoder.transformer_decoder.layers.named_parameters():
			self.writer.add_histogram(name, parameters, epoch)

	def add_loss(self,epoch:int,loss:int):
		self.writer.add_scalar('Loss', loss, epoch)

	def add_bleu(self,epoch:int,bleu:int):
		self.writer.add_scalar('Bleu', bleu, epoch)

	def add_images(self,epoch:int,images:Tensor,expected_caption:str,generated_caption:str):
		px = 1/plt.rcParams['figure.dpi']  # pixel in inches
		rows = int(sqrt(len(images)))
		hfont = {'fontname':'Helvetica'}
		fig, ax = plt.subplots(rows,rows,figsize=(1500*px, 1500*px))
		fig.tight_layout(pad=10)
		image = None
		for row in range(rows):
			for col in range(rows):
				index = (row*rows)+col
				image = images[index].detach().clone()
				image = Visualization.process_image(image)
				expected, generated = "\n".join(wrap(expected_caption[index])), "\n".join(wrap(generated_caption[index]))
				title = f'Expected:\n\n{expected} \n\n Generated: \n\n{generated}'
				ax[row][col].get_xaxis().set_visible(False)
				ax[row][col].get_yaxis().set_visible(False)
				ax_title = ax[row][col].set_title(title,**hfont)
				ax_title.set_y(1.05)
				ax[row][col].imshow(image)

		self.writer.add_figure('Captions comparison',fig, close=True, global_step=epoch)
		plt.close()

tensorboard_panel = Tensorboard()
tensorboard_panel_eval = Tensorboard('runs/ImageCaptioningEvaluate')
