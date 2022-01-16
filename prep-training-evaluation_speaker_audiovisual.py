# Importin modules
import numpy as np
import pandas as pd
import pickle
import argparse
import time
import json
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

# MODEL + SUPPORT FUNCTIONS FOR TRAINING
# Multimodal model
class MM(nn.Module): 
	def __init__(self):
		super().__init__()
		self.conv1_a = nn.Conv2d(1, 128, kernel_size=(10, 23), stride=3)
		
		self.conv1_v = nn.Conv2d(8, 16, 5, 1, 1)
		self.conv2_v = nn.Conv2d(16, 32, 5, 1, 1)
		self.conv3_v = nn.Conv2d(32, 64, 5, 1, 1)
		
		self.drop_a = nn.Dropout(p=0.7)
		self.drop_v = nn.Dropout(p=0.8)
		self.drop_x = nn.Dropout(p=0.7)
		#---- trying to find out what the input shape to fc1 is ----
		a = torch.randn(max_len_features, 23).view(-1, 1, max_len_features, 23) #random tensor of correct shape
		self._to_linear_a = None
		self.convs_a(a)
		v = torch.randn(8,100,100).view(-1, 8, 100, 100)
		self._to_linear_v = None
		self.convs_v(v)
		#-----------------------------------------------------------
		self.fc1 = nn.Linear(self._to_linear_a+self._to_linear_v, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc_out = nn.Linear(128, 4)
	
	def convs_a(self, a):
		a = F.relu(self.conv1_a(a)) 
		a = torch.squeeze(a, dim=3) # gets rid of the last dimension (torch.Size([a, b, c, 1]) -> torch.Size([a, b, c]))
		a = F.max_pool1d(a, 5)
		a = self.drop_a(a)
		if self._to_linear_a is None:
			self._to_linear_a = np.prod(a[0].shape) # catching the output (x[0].shape[0]*x[0].shape[1]*x[0].shape[2]) 
			print("self._to_linear_a", self._to_linear_a)
		return a
	
	def convs_v(self, v):
		v = F.max_pool2d(F.relu(self.conv1_v(v)), (2,2))
		v = F.max_pool2d(F.relu(self.conv2_v(v)), (2,2))
		v = F.max_pool2d(F.relu(self.conv3_v(v)), (2,2))
		v = self.drop_v(v)
		if self._to_linear_v is None:
			self._to_linear_v = np.prod(v[0].shape) # catching the output (x[0].shape[0]*x[0].shape[1]*x[0].shape[2]) 
			print("self._to_linear_v", self._to_linear_v)
		return v
	
	def forward(self, a, v):
		a = self.convs_a(a) # passing through convolutional layers
		a = a.view(-1, self._to_linear_a) # basically flattening
		v = self.convs_v(v) # passing through convolutional layers
		v = v.view(-1, self._to_linear_v) # basically flattening
		
		# Combining the outputs
		x = torch.cat((a, v), dim=1) #sizes must match except in the dimension of the concatenation
		x = F.relu(self.fc1(x))
		x = self.drop_x(x)
		x = F.relu(self.fc2(x))
		x = self.drop_x(x)
		x = self.fc_out(x)
		
		return x

# This trains the model
def training(X_train_a, X_train_v, y_train):
	net.train()
	train_batch_loss = []
	train_correct = 0
	train_total = 0
	for i in range(0, len(X_train_a), BATCH_SIZE):
		X_train_batch_a = X_train_a[i:i+BATCH_SIZE].view(-1, 1, max_len_features, 23)
		X_train_batch_v = X_train_v[i:i+BATCH_SIZE].view(-1, 8, 100, 100)
		y_train_batch = y_train[i:i+BATCH_SIZE]
		# Fitment (zeroing the gradients)
		optimizer.zero_grad()
		train_outputs = net(X_train_batch_a, X_train_batch_v)
		for j, k in zip(train_outputs, y_train_batch):
			if torch.argmax(j) == k:
				train_correct += 1
			train_total += 1
		train_loss = loss_function(train_outputs, y_train_batch)
		train_batch_loss.append(train_loss.item())
		
		train_loss.backward()
		optimizer.step()
		
	train_loss_epoch = round(float(np.mean(train_batch_loss)),4) # over all BATCHES
	train_acc_total = round(train_correct/train_total, 4) # over all FILES
	
	return train_loss_epoch, train_acc_total

# This tests the model (dev/final_test)
def testing(X_a, X_v, y, final_test=False):
	net.eval()
	correct = 0
	total = 0
	batch_loss = []
	final_test_predictions = []
	with torch.no_grad():
		for i in range(0, len(X_a), BATCH_SIZE):
			X_batch_a = X_a[i:i+BATCH_SIZE].view(-1, 1, max_len_features, 23)
			X_batch_v = X_v[i:i+BATCH_SIZE].view(-1, 8, 100, 100)
			y_batch = y[i:i+BATCH_SIZE]
			outputs = net(X_batch_a, X_batch_v)
			if final_test:
				final_test_predictions.append(torch.argmax(outputs, dim=1))
			loss = loss_function(outputs, y_batch)
			batch_loss.append(loss.item())
			for j, k in zip(outputs, y_batch):
				if torch.argmax(j) == k:
					correct += 1
				total += 1
	loss_epoch = round(float(np.mean(batch_loss)),4) # over all BATCHES
	acc_total = round(correct/total, 4) # over all FILES
	
	if final_test:
		return torch.cat(final_test_predictions), acc_total
	else:
		return loss_epoch, acc_total

# Argparse constructor
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=True,
	help = "path to the dataset file")
parser.add_argument("-l", "--label", required=True,
	help = "type of label to be used (LABEL or ACTIVATION or VALENCE")
args = vars(parser.parse_args())

# Converts continuous labels (activation/valence) into discrete classes
def map_to_bin(cont_label):
	if cont_label <= 2.5:
		return 0.0
	elif 2.5 < cont_label < 3.5:
		return 1.0
	elif cont_label >= 3.5:
		return 2.0

# Splits data according to the 6 original sessions (given the speaker id)
def create_sessions(df):
	# Audio features
	a_1 = []
	a_2 = []
	a_3 = []
	a_4 = []
	a_5 = []
	a_6 = []
	# Video features
	v_1 = []
	v_2 = []
	v_3 = []
	v_4 = []
	v_5 = []
	v_6 = []
	# Labels (category/activation/valnece depending on the parsed argument)
	l_1 = []
	l_2 = []
	l_3 = []
	l_4 = []
	l_5 = []
	l_6 = []
	for i in df.index:
		session = i[17:19] #contains session nr
		if session == "01":
			a_1.append(df.loc[i,"AUDIO"])
			v_1.append(df.loc[i,"VIDEO"])
			l_1.append(df.loc[i,args["label"]])
		elif session == "02":
			a_2.append(df.loc[i,"AUDIO"])
			v_2.append(df.loc[i,"VIDEO"])
			l_2.append(df.loc[i,args["label"]])
		elif session == "03":
			a_3.append(df.loc[i,"AUDIO"])
			v_3.append(df.loc[i,"VIDEO"])
			l_3.append(df.loc[i,args["label"]])
		elif session == "04":
			a_4.append(df.loc[i,"AUDIO"])
			v_4.append(df.loc[i,"VIDEO"])
			l_4.append(df.loc[i,args["label"]])
		elif session == "05":
			a_5.append(df.loc[i,"AUDIO"])
			v_5.append(df.loc[i,"VIDEO"])
			l_5.append(df.loc[i,args["label"]])
		elif session == "06":
			a_6.append(df.loc[i,"AUDIO"])
			v_6.append(df.loc[i,"VIDEO"])
			l_6.append(df.loc[i,args["label"]])
		else:
			print(f'ERROR occured for: {i}')
	
	return [a_1, a_2, a_3, a_4, a_5, a_6], [v_1, v_2, v_3, v_4, v_5, v_6], [l_1, l_2, l_3, l_4, l_5, l_6]
	
# SUPPORT FUNCTIONS FOR THE K-FOLD-SPLIT (SESSION-WISE-SPLIT)

# K-fold functions (AUDIO)
# Concatenating with zeros/cutting to length
def zeros(d, m):
	start_time = time.time()
	f = torch.stack( # concatenating along a NEW dim
		[torch.cat( # concatenating along one of GIVEN dims
		[
			torch.Tensor(i[:m]), # takes SLICE of i from 0 to given NR
			torch.zeros((
				(m - i.shape[0]) if m > i.shape[0] else 0, i.shape[1])) # i.shape[1] can be set to a constant since: nr of extracted features
		], dim=0)
			for i in d], dim=0)
	end_time = time.time()
	duration = end_time - start_time
	print(f'processing took {duration} seconds')
	return f

# Extracts mean and std over all the data
def mean_std(features):
	c_features = np.concatenate(features, axis=0)
	features_mean, features_std = np.mean(c_features), np.std(c_features, ddof=0)
	
	return features_mean, features_std

# Creats list of arrays (since each array has a different shape due to audiolength)
def list_arrays(x):
	listed = []
	for i in x:
		for j in i:
			listed.append(j)
	
	return listed

# K-fold functions (BOTH)
# Standardization
def standardize(features, mean, std, video=False):
	features = (features - mean) / (std + 0.0000001) # adding epsilon to avoid errors (e.g. division by 0)
	if video:
		return torch.Tensor(features)
	else:
		return features

# Splits validation set into dev and test (least populated class needs to have AT LEAST 2 MEMBERS)
def SSS(X_val_a, X_val_v, y_val):
	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
	sss.get_n_splits(X_val_a, y_val)
	for train_index, test_index in sss.split(X_val_a, y_val):
		X_dev_a, X_final_test_a = X_val_a[train_index], X_val_a[test_index]
		X_dev_v, X_final_test_v = X_val_v[train_index], X_val_v[test_index]
		y_dev, y_final_test = y_val[train_index], y_val[test_index]
		
		return X_dev_a, X_final_test_a, X_dev_v, X_final_test_v, y_dev, y_final_test

# Provides insights into the data/predictions
def insight(actual, pred):
	total = 0
	actual_dict = {0:0, 1:0, 2:0, 3:0}
	pred_dict = {0:0, 1:0, 2:0, 3:0}
	correct_dict = {0:0, 1:0, 2:0, 3:0} 
	for a, p in zip(actual, pred):
		actual_dict[int(a)] +=1
		pred_dict[int(p)] +=1
		if a == p:
			correct_dict[int(p)] +=1
		total += 1
	print("ACTUAL:")
	for i in actual_dict:
		print(i, '\t', actual_dict[i], '\t', round(actual_dict[i]/total*100, 4), '%')
	print("PREDICTED:")
	for i in pred_dict:
		print(i, '\t', pred_dict[i], '\t', round(pred_dict[i]/total*100, 4), '%')
	print("CORRECT:")
	for i in correct_dict:
		if actual_dict[i] == 0:
			print(0)
		else:
			print(i, '\t', correct_dict[i], '\t', round(correct_dict[i]/actual_dict[i]*100, 4), '%')

if __name__ == '__main__':
	# Loading pandas dataset
	df = pd.read_pickle(args["data"])

	# Converts continuous labels (activation/valence) into discrete classes
	for i in enumerate(df.index):
		df.at[i[1], 'ACTIVATION'] = map_to_bin(df['ACTIVATION'][i[0]])
		df.at[i[1], 'VALENCE'] = map_to_bin(df['VALENCE'][i[0]])

	# Converting emotion labels into classes
	df["LABEL"].replace({'anger': 0, 'happiness': 1, 'neutral': 2, 'sadness': 3}, inplace=True)

	# Computing the maximal length of frames (mean_length + std)
	nr_of_frames = [i.shape[0] for i in df["AUDIO"]]
	max_len_features = int(np.mean(nr_of_frames)+np.std(nr_of_frames))
	print(f'features will be cut/extended to length: {max_len_features}')

	# Splitting data according to the 6 original sessions (given the speaker id)
	a, v, l = create_sessions(df)

	# Batch size and nr of epochs
	BATCH_SIZE = 128
	EPOCHS = 25

	# 6-FOLD CROSS VALIDATION
	statistics = {}
	accuracy_of_k_fold = []
	F1u_of_k_fold = []
	F1w_of_k_fold = []
	for i in range(len(l)):
		print(f'Iteration: {i}')
		X_train_a, X_test_a = list_arrays(a[:i] + a[i+1:]), a[i]
		X_train_v, X_test_v = np.concatenate((v[:i] + v[i+1:]),axis=0), np.array(v[i])
		y_train, y_test = np.concatenate((l[:i] + l[i+1:]),axis=0), np.array(l[i])
		class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
		print(f'CLASS WEIGHTS: {class_weights}')
		
		features_mean_a, features_std_a = mean_std(X_train_a) #concatenates all timeframes into one array to extract mean+std (necessary sonce files have dofferent nr of frames)
		features_mean_v, features_std_v = np.mean(X_train_v), np.std(X_train_v)
			
		# Standardizing the features of the sessions individually
		X_train_a = [standardize(i, features_mean_a, features_std_a) for i in X_train_a]
		X_train_v = standardize(X_train_v, features_mean_v, features_std_v, video=True) #standardizing the features of the session-combinations
		# Cutting/padding TRAIN to uniform length
		X_train_a = zeros(X_train_a, max_len_features)
		
		# Standardizing TEST with MEAN & STD of TRAIN
		X_test_a = [standardize(i, features_mean_a, features_std_a) for i in X_test_a]
		X_test_v = standardize(X_test_v, features_mean_v, features_std_v, video=True)
		# Cutting/padding TEST to uniform length
		X_test_a = zeros(X_test_a, max_len_features)
		
		# Splitting TEST into DEV and FINAL_TEST
		X_dev_a, X_final_test_a, X_dev_v, X_final_test_v, y_dev, y_final_test = SSS(X_test_a, X_test_v, y_test)
		
		# Gathering general information
		l_t, c_t = np.unique(y_train, return_counts=True)
		l_d , c_d = np.unique(y_dev, return_counts=True)
		l_f_t, c_f_t = np.unique(y_final_test, return_counts=True)
		
		general = {"train_total": len(y_train), "train_dist": c_t.tolist(), 
				"dev__total": len(y_dev), "dev__dist": c_d.tolist(), 
				"final_test__total": len(y_final_test), "final_test__dist": c_f_t.tolist()}
		
		# Converting labels and class_weights to tensors
		y_train = torch.LongTensor(y_train)
		y_dev = torch.LongTensor(y_dev).cuda()
		y_final_test = torch.LongTensor(y_final_test).cuda()
		class_weights = torch.Tensor(class_weights).cuda()
		
		X_dev_a = X_dev_a.cuda()
		X_dev_v = X_dev_v.cuda()
		X_final_test_a = X_final_test_a.cuda()
		X_final_test_v = X_final_test_v.cuda()
		
		# Performing random permutation
		perm_ind = torch.randperm(len(y_train))
		X_train_a = X_train_a[perm_ind].cuda()
		X_train_v = X_train_v[perm_ind].cuda()
		y_train = y_train[perm_ind].cuda()
		
		print(f' X_train_a shape is: {X_train_a.shape} y_train length is: {len(y_train)}')
		print(f' X_train_v shape is: {X_train_v.shape} y_train length is: {len(y_train)}')
		
		print(f' X_dev_a shape is: {X_dev_a.shape} y_dev length is: {len(y_dev)}')
		print(f' X_dev_v shape is: {X_dev_v.shape} y_dev length is: {len(y_dev)}')
		
		print(f' X_final_test_a shape is: {X_final_test_a.shape} y_final_test length is: {len(y_final_test)}')
		print(f' X_final_test_v shape is: {X_final_test_v.shape} y_final_test length is: {len(y_final_test)}')
		
		#-----------TRAINING STEP--------------
		net = MM().cuda()  # reinitializing the NN for the new fold (in order to get rid of the learned parameters)
		
		optimizer = optim.Adam(net.parameters(), lr= 0.0001)
		loss_function = nn.CrossEntropyLoss(weight=class_weights)
		
		fold = {"general": general, "train_loss_fold": [], "train_acc_fold": [], "dev_loss_fold": [], "dev_acc_fold": []}
		for epoch in range(EPOCHS):
			# Training
			train_loss_epoch, train_acc_epoch = training(X_train_a, X_train_v, y_train)
			fold["train_loss_fold"].append(train_loss_epoch)
			fold["train_acc_fold"].append(train_acc_epoch)
			# Evaluation on DEV
			dev_loss_epoch, dev_acc_epoch = testing(X_dev_a, X_dev_v, y_dev)
			fold["dev_loss_fold"].append(dev_loss_epoch)
			fold["dev_acc_fold"].append(dev_acc_epoch)
			print(f'loss: {train_loss_epoch} {dev_loss_epoch} acc: {train_acc_epoch} {dev_acc_epoch}')
		
		# Evaluation on FINAL_TEST
		final_test_predictions, final_test_acc_total = testing(X_final_test_a, X_final_test_v, y_final_test, final_test=True)
		fold["ACC"] = final_test_acc_total
		accuracy_of_k_fold.append(final_test_acc_total)
		print(f'Accuracy of the final test: {final_test_acc_total}%')
		
		F1u = round(f1_score(torch.clone(y_final_test).cpu(), torch.clone(final_test_predictions).cpu(), average='macro'),4) #average='macro'
		fold["F1u"] = F1u
		F1u_of_k_fold.append(F1u)
		print(f'F1u-Score of the final test: {F1u}')
		
		F1w = round(f1_score(torch.clone(y_final_test).cpu(), torch.clone(final_test_predictions).cpu(), average='weighted'),4) #average='macro' average='weighted'
		fold["F1w"] = F1w
		F1w_of_k_fold.append(F1w)
		print(f'F1w-Score of the final test: {F1w}')
		
		fold["y_final_test"] = y_final_test.cpu().tolist()
		fold["final_test_predictions"] = final_test_predictions.cpu().tolist()
		
		print(y_final_test[:20])
		print(final_test_predictions[:20])
		insight(y_final_test, final_test_predictions)
		statistics[i] = fold
		print('\n')

	statistics["total_ACC"] = round(np.mean(accuracy_of_k_fold),4)
	statistics["total_F1u"] = round(np.mean(F1u_of_k_fold),4)
	statistics["toal_F1w"] = round(np.mean(F1w_of_k_fold),4)
	statistics["max_len_audio"] = max_len_features
	statistics["batch_size"] = BATCH_SIZE 
	statistics["epochs"] = EPOCHS 

	print(f'AVERAGE ACCURACY OVER FOLDS IS: {round(np.mean(accuracy_of_k_fold),4)}%')
	print(f'AVERAGE F1u OVER FOLDS IS: {round(np.mean(F1u_of_k_fold),4)}')
	print(f'AVERAGE F1w OVER FOLDS IS: {round(np.mean(F1w_of_k_fold),4)}')

	aff = input("store the data (y/n): ")
	if aff == "y":
		with open('stats_mm_'+args["label"]+'_f'+'.json', 'w', encoding='utf-8') as f:
			json.dump(statistics, f, ensure_ascii=False, indent=2)
