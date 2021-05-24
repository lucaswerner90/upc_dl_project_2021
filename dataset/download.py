import argparse
import os
import json

cwd = os.getcwd()

class DownloadDataset():
	@staticmethod
	def download(filepath:str) -> None:
		data = None
		with open(os.path.join(cwd,'dataset','kaggle.json'),'r') as f:
			data = json.load(f)

		os.environ['KAGGLE_USERNAME'] = data['username']
		os.environ['KAGGLE_KEY'] = data['key']

		# https://github.com/Kaggle/kaggle-api
		os.system(f'kaggle datasets download adityajn105/flickr8k --path {filepath} --unzip')
	
if __name__ == '__main__':
	print('Downloading Flickr8k dataset...')
	filepath = os.path.join(cwd,'data')
	DownloadDataset.download(filepath)
