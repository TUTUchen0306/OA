import numpy as np
import matplotlib.pyplot as plt
import os
import json

dirlist = ['20221122-153553_SVRG_MNIST_logistic', '20221125-165740_SVRG_MNIST_logistic', '20221122-222600_SVRG_MNIST_logistic']
acc_low=0.9

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,6), sharey='row')

axes[0][0].set_title("Training Loss")
axes[0][1].set_title("Validation Loss")
axes[1][0].set_title("Training Accuracy")
axes[1][1].set_title("Validation Accuracy")

for directory in dirlist:

	dirpath = os.path.join("outputs", directory)
	print(os.path.join(dirpath, "args.json"))
	with open(os.path.join(dirpath, "args.json"), 'r') as f:
		args = json.load(f)
		
		data = np.load(os.path.join(dirpath, 'train_stats.npz'))
		train_loss = data['train_loss']
		train_acc = data['train_acc']
		val_loss = data['val_loss']
		val_acc = data['val_acc']

		axes[0][0].plot(train_loss, label=f"{args['optimizer']}: {args['lr']}, {args['batch_size']}, {args['nn_model']}")
		axes[0][1].plot(val_loss, label=f"{args['optimizer']}: {args['lr']}, {args['batch_size']}, {args['nn_model']}")
		axes[1][0].plot(train_acc, label=f"{args['optimizer']}: {args['lr']}, {args['batch_size']}, {args['nn_model']}")
		axes[1][1].plot(val_acc, label=f"{args['optimizer']}: {args['lr']}, {args['batch_size']}, {args['nn_model']}")

axes[1][0].set_ylim(acc_low, 1)
plt.tight_layout()
axes[0][0].legend(loc=0)
axes[0][1].legend(loc=0)
axes[1][0].legend(loc=0)
axes[1][1].legend(loc=0)
plt.savefig('myplot.png')