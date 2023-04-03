import gurobipy as gp
import numpy as np
import scipy
import copy
from scipy import io
import matplotlib.pyplot as plt

def cal_loss(data, x):
	data_len = len(data[0])
	loss = 0
	for a in data:
		tmp = sum(a[i] * x[i] for i in range(data_len))
		loss += -np.log(tmp)

	return loss

def build():
	m = gp.Model()
	x = m.addVars(500, lb=0.0, vtype='S', name="X")
	m.addConstr(gp.quicksum(x[i] for i in range(500)) == 1)
	m.update()
	m.optimize()
	return m

def cal_direction(data, model, x):
	dot_list = []

	data_len = len(data[0])

	for a in data:
		tmp = sum(a[i] * x[i] for i in range(data_len))
		dot_list.append(tmp)

	all_vars = model.getVars()


	grad = []

	for i in range(data_len):
		tmp = 0.0

		for j in range(len(data)):
			if dot_list[j] == 0:
				continue
			else:
				tmp += data[j][i] / dot_list[j]

		grad.append(-tmp)

	model.setObjective(gp.quicksum(all_vars[i] * grad[i] for i in range(len(grad))), sense=gp.GRB.MINIMIZE)
	model.optimize()

	return

if __name__ == "__main__":
	file_name = '473500_wk.mat'
	data = scipy.io.loadmat(file_name)
	data = data['W']

	iteration = 400

	model = build()
	model.Params.LogToConsole = 0
	last_x = []
	loss_a = []
	now_min = 10
	total_min = 10
	group = 30
	for i in range(iteration):
		learning_rate = 2.0 / (i + 2)
		old_x = []
		if i == 0:
			for _ in range(500):
				old_x.append(1/500)
		else:
			old_x = copy.deepcopy(last_x)

		cal_direction(data, model, old_x)

		y = model.getVars()
		new_x = []

		for j in range(len(y)):
			new_x.append(old_x[j] * (1 - learning_rate) + y[j].X * learning_rate)

		loss = cal_loss(data, new_x)
		now_min = min(now_min, loss)
		total_min = min(total_min, loss)
		if i == 0:
			loss_a.append(loss)
		if (i+1) % group == 0 and i < group*7:
			loss_a.append(now_min)
			now_min = 10

		last_x = copy.deepcopy(new_x)

	for i in range(len(loss_a)):
		loss_a[i] -= total_min
		print(f'iteration {i} : {loss_a[i]}')


	plt.plot(loss_a)
	plt.ylim([0.0000000001,1.0])
	plt.yscale('log')
	plt.show()




