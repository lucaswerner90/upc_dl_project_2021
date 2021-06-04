from torch.functional import Tensor
from torch.utils.tensorboard import SummaryWriter
from model.visualization import Visualization

import matplotlib.pyplot as plt

class Tensorboard:
	writer = SummaryWriter('runs/ImageCaptioning')

	def add_sentences_comparison(self,epoch:int,expected:str,generated:str):
		self.writer.add_text('Expected/Generated', f'Expected: {expected} \n\n Generated:{generated}', epoch)

	def add_loss(self,epoch:int,loss:int):
		self.writer.add_scalar('Loss/train', loss, epoch)

	def add_bleu(self,epoch:int,bleu:int):
		self.writer.add_scalar('Bleu/train', bleu, epoch)

	def add_image(self,epoch:int,image:Tensor,expected_caption:str,generated_caption:str):
		image = Visualization.process_image(image)
		plt.tight_layout()
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.set_title(f'Exp: {expected_caption}\n Gen: {generated_caption}',fontsize=8)
		ax.imshow(image)
		self.writer.add_figure('Generated captions',fig, global_step=epoch)


tensorboard_panel = Tensorboard()
