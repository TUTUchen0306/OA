import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

def zero_model(model, data):
	return model(data)

def one_model(model, data, avg_weight, avg_bias, times):
	new_weight = model.weight + avg_weight * times
	new_bias = model.bias + avg_bias * times

	new_weight /= (times+1)
	new_bias /= (times+1)

	out = model(data)

	return out, F.linear(data, new_weight, new_bias), new_weight, new_bias


def half_avg_model(model, data, avg_weight, avg_bias, weight_list, bias_list, times):
	out = model(data)

	if times == 0:
		return out, F.linear(data, model.weight, model.bias), model.weight, model.bias
	elif times == 1:
		weight_list.pop(0)
		bias_list.pop(0)
		return out, F.linear(data, model.weight, model.bias), model.weight, model.bias
	else:
		n = int((times+2) / 2)
		if(times % 2 == 0):
			new_weight = model.weight + (avg_weight * (n-1))
			new_bias = model.bias + (avg_bias * (n-1))
			new_weight /= n
			new_bias /= n
			return out, F.linear(data, new_weight, new_bias), new_weight, new_bias
		else:
			new_weight = model.weight + (avg_weight * n - weight_list[0])
			new_bias = model.bias + (avg_bias * n - bias_list[0])
			new_weight /= n
			new_bias /= n
			weight_list.pop(0)
			bias_list.pop(0)
			return out, F.linear(data, new_weight, new_bias), new_weight, new_bias

def D_model(model, data, avg_weight, avg_bias, times):

	new_weight = model.weight * 0.5 + avg_weight * 0.5
	new_bias = model.bias * 0.5 + avg_bias * 0.5

	out = model(data)

	return out, F.linear(data, new_weight, new_bias), new_weight, new_bias

def W_model(model, data, avg_weight, avg_bias, times):
	p = 2 / (times+2)
	new_weight = model.weight * p + avg_weight * (1-p)
	new_bias = model.bias * p + avg_bias * (1-p)

	out = model(data)


	return out, F.linear(data, new_weight, new_bias), new_weight, new_bias


def W2_model(model, data, avg_weight, avg_bias, times):
	p = 6 * (times+1) / ((2 * times + 3)*(times + 2))
	new_weight = model.weight * p + avg_weight * (1-p)
	new_bias = model.bias * p + avg_bias * (1-p)

	out = model(data)

	return out, F.linear(data, new_weight, new_bias), new_weight, new_bias