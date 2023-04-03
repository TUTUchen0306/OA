import argparse
import numpy as np
import torch
import os
import json
from datetime import datetime
import time
from torch import nn
from torch import optim
from torch.nn import functional as F
from util import HingeLoss, read_file
from model_type import zero_model, one_model, half_avg_model, D_model, W_model, W2_model

parser = argparse.ArgumentParser(description="implement O(1/t) PSGD.")
parser.add_argument("--dataset", type=str, default="covtype.libsvm.binary", help="dataset file name")
parser.add_argument("--p", type=int, default=54, help="number of feature")
parser.add_argument('--n_iter', type=int, default=30, help="number of training iterations")
parser.add_argument('--model', type=str, default="0", help="number of training iterations")
parser.add_argument('--lr', type=str, default="tmu1", help="number of training iterations")



OUTPUT_DIR = "outputs"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Using device: {}".format(device))



def train(X, Y, model, avg_model, avg_model_type, avg_weight, avg_bias, loss_fn, lr_type, iteration, weight_list=[], bias_list=[]):
	n = len(Y)
	X = torch.FloatTensor(X)
	Y = torch.FloatTensor(Y)

	mu = 1 / n

	optimizer = optim.SGD(model.parameters(), lr=1/n)

	model.train()

	accuracy, avg_loss = 0, 0

	idx_array = np.random.permutation(n)

	nt = iteration * n

	for times, idx in enumerate(idx_array):

		optimizer.zero_grad()

		data, label = X[idx], Y[idx]

		if lr_type == "tmu1":
			optimizer.param_groups[0]['lr'] = n / (nt + times + 1)
		else:
			optimizer.param_groups[0]['lr'] = 2 * n / (nt + times + 1)


		data = data.to(device)
		label = label.to(device)
		yhat = 0


		avg_weight = avg_weight.to(device)
		avg_bias = avg_bias.to(device)

		old_state = model.state_dict()

		if avg_model_type == "0":
			yhat = avg_model(model, data)
		elif avg_model_type == "0.5":
			yhat, avg_out, avg_weight, avg_bias = avg_model(model, data, avg_weight, avg_bias, weight_list, bias_list, nt+times)
			weight_list.append(model.weight)
			bias_list.append(model.bias)
		else:
			yhat, avg_out, avg_weight, avg_bias = avg_model(model, data, avg_weight, avg_bias, nt+times)


		loss_iter = loss_fn(yhat, label)

		l2_reg = torch.tensor([0.])

		l2_reg = l2_reg.to(device)

		for param in model.parameters():
		    l2_reg[0] += (param ** 2).sum()

		loss_iter += 1/2 * mu * l2_reg[0]

		loss_iter.backward()
		
		optimizer.step()

		if avg_model_type == "0":

			if yhat > 0:
				yhat = 1
			else:
				yhat = -1
			
			if yhat == label:
				accuracy += 1

			avg_loss += loss_iter.data.item()

		else:
			loss_rec = loss_fn(avg_out, label)

			loss_rec += 1/2 * mu * l2_reg[0]

			if avg_out > 0:
				avg_out = 1
			else:
				avg_out = -1
			

			if avg_out == label:
				accuracy += 1

			avg_loss += loss_rec.data.item()

	avg_loss /= n
	accuracy /= n

	return avg_loss, avg_weight, avg_bias, accuracy

if __name__ == "__main__":

	args = parser.parse_args()
	args_dict = vars(args)
    
	dataset = args.dataset
	p = args.p
	n_iter = args.n_iter
	avg_type = args.model
	lr_type = args.lr

	avg_weight, avg_bias = torch.zeros(p), torch.zeros(1)


	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	model_name = timestamp + "_" + args.lr + "_" + args.model
	log_dir = os.path.join(OUTPUT_DIR, model_name)
	if not os.path.isdir(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	if not os.path.isdir(log_dir):
		os.mkdir(log_dir)
	with open(os.path.join(log_dir, "args.json"), "w") as f:
		json.dump(args_dict, f)

	data, label = read_file(dataset, p)

	model = nn.Linear(p, 1)
	model.to(device)

	loss_fn = HingeLoss()

	loss_arr = []
	weight_list, bias_list = [], []

	for iteration in range(n_iter):
		loss, accuracy = 0, 0
		if avg_type == "0":
			loss, avg_weight, avg_bias, accuracy = train(data, label, model, zero_model, avg_type, avg_weight, avg_bias, loss_fn, lr_type, iteration)
		elif avg_type == "1":
			loss, avg_weight, avg_bias, accuracy = train(data, label, model, one_model, avg_type, avg_weight, avg_bias, loss_fn, lr_type, iteration)
		elif avg_type == "0.5":
			loss, avg_weight, avg_bias, accuracy = train(data, label, model, half_avg_model, avg_type, avg_weight, avg_bias, loss_fn, lr_type, iteration, weight_list, bias_list)
		elif avg_type == "D":
			loss, avg_weight, avg_bias, accuracy = train(data, label, model, D_model, avg_type, avg_weight, avg_bias, loss_fn, lr_type, iteration)
		elif avg_type == "W":
			loss, avg_weight, avg_bias, accuracy = train(data, label, model, W_model, avg_type, avg_weight, avg_bias, loss_fn, lr_type, iteration)
		elif avg_type == "W2":
			loss, avg_weight, avg_bias, accuracy = train(data, label, model, W2_model, avg_type, avg_weight, avg_bias, loss_fn, lr_type, iteration)
		
		loss_arr.append(loss)

		print("iteration {} : now loss => {:.10f}, now accuracy => {:.5f}".format(iteration, loss, accuracy))

	np.savez(os.path.join(log_dir, 'train_stats.npz'), loss=np.array(loss_arr))
