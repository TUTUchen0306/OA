import numpy as np
import matplotlib.pyplot as plt
import os
import json

dirlist = ['20221125-160640_SGD_MNIST_logistic', '20221125-165740_SVRG_MNIST_logistic']
optimal = 1e5


for directory in dirlist:

	dirpath = os.path.join("outputs", directory)
	print(os.path.join(dirpath, "args.json"))
	with open(os.path.join(dirpath, "args.json"), 'r') as f:
		args = json.load(f)
		
		data = np.load(os.path.join(dirpath, 'train_stats.npz'))
		train_loss = data['train_loss']
		if(optimal > min(train_loss)):
			optimal = min(train_loss)


for directory in dirlist:
	nl = []
	dirpath = os.path.join("outputs", directory)
	print(os.path.join(dirpath, "args.json"))
	with open(os.path.join(dirpath, "args.json"), 'r') as f:
		args = json.load(f)
		
		data = np.load(os.path.join(dirpath, 'train_stats.npz'))
		train_loss = data['train_loss']
		train_loss -= optimal
		for i in train_loss:
			cnt = 0
			while(i < 1):
				cnt -= 1
				i *= 10
				if cnt < -20:
					break
			nl.append(cnt)
		plt.plot(np.array(nl), label=f"{args['optimizer']}: {args['lr']}")
plt.legend(loc=0)
plt.savefig('dif.png')