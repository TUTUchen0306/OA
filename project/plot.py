import os
import json
import numpy as np
import matplotlib.pyplot as plt

dirlist = ['20221227-004510_tmu1_0', '20221226-135205_tmu1_1', '20221226-184931_tmu1_W', '20221228-104545_tmu1_W2', '20221227-201948_tmu2_W']
title = ''
for directory in dirlist:
	dirpath = os.path.join("outputs", directory)
	print(os.path.join(dirpath, "args.json"))
	with open(os.path.join(dirpath, "args.json"), 'r') as f:
		args = json.load(f)
		
		title = args['dataset']

		data = np.load(os.path.join(dirpath, 'train_stats.npz'))
		loss = data['loss']

		lr_type = ''
		if args['lr'] == "tmu1":
			lr_type = "1/tmu"
		else:
			lr_type = "2/(t+1)mu"

		plt.plot(loss, label=f"({lr_type}, {args['model']})")


plt.xlabel('Effective Passes')
plt.ylabel('Objective')
plt.ylim([0.5,1.0])
plt.yscale('log')
plt.legend(loc=0)
plt.title(f'{title}')
plt.tight_layout()
plt.savefig('myplot.png')
plt.close()