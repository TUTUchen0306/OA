import torch
from torch import nn
import numpy as np


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, inputs, target):
        L = (1 - target * inputs).clamp(min=0)
        return torch.mean(L)

def read_file(file_name, p):
	x = []
	y = []
	
	f = open(file_name, 'r')
	lines = f.readlines()
	
	for line in lines:
		features = np.zeros(p, dtype='float16')
		data_list = line.strip().split() 

		for i in range(len(data_list)):
			if i == 0:
				if int(data_list[0]) == 1:
					y.append(1)
				else:
					y.append(-1)
			else:
				data = data_list[i].split(':')
				features[int(data[0])-1] = float(data[1])
		x.append(features)

	return np.array(x), np.array(y)