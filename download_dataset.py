import argparse

if __name__ == '__main__':
	print('Downloading dataset...')
	parser = argparse.ArgumentParser()
	parser.add_argument("--url",help="specifies the URL to download the dataset",default='', type=str)
	args = parser.parse_args()
	print(args.url)
