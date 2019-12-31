#!/usr/bin/env python3

import csv
import os
import random
import time

import easyargs
import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch import nn, optim
from torch.optim import lr_scheduler

matplotlib.use('Agg')
# from create_dataset_cifar100 import myDataset
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
os.sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from glo.utils import save_checkpoint, get_cub_param, get_loader_with_idx, get_cifar100_small, get_classifier, \
	get_test_stl_param, get_train_n_unlabled_stl_param
from cub2011 import Cub2011

from glo.evaluation import fewshot_setting, accuracy
from logger import Logger

dim = 512


def train_linear_classifier(model, criterion, optimizer, train_data, num_epochs, seed, name, offset_idx=0,
                            offset_label=0, batch_size=128, fewshot=False, augment=False, autoaugment=False,
                            start_epoch=0, aug_param=None, test_data=None):
	if aug_param is None:
		aug_param = {'std': None, 'mean': None, 'rand_crop': 32, 'image_size': 32}
	model.train()
	sampler = None
	milestones = [60, 120, 160]
	if fewshot:
		num_epoch_repetition = fewshot_setting(train_data)
		num_epochs *= num_epoch_repetition
		milestones = list(map(lambda x: num_epoch_repetition * x, milestones))
	train_data_loader = get_loader_with_idx(train_data, batch_size, num_workers=8,
	                                        augment=augment,
	                                        offset_idx=offset_idx, offset_label=offset_label, sampler=sampler,
	                                        autoaugment=autoaugment, **aug_param)
	# train_data_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=4)
	print(f" train_len:{len(train_data)}")
	if aug_param['image_size'] > 32:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
	else:
		scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
	train_loss = 0
	correct = 0
	total = 0
	use_cuda = torch.cuda.is_available()
	for epoch in range(start_epoch, num_epochs):
		model.train()
		end_epoch_time = time.time()
		for i, (_, inputs, targets) in enumerate(train_data_loader):
			if use_cuda:
				inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
			optimizer.zero_grad()
			output = model(inputs)
			# _, preds = torch.max(output.data, 1)
			loss = criterion(output, targets)
			loss.backward()
			optimizer.step()

			train_loss += loss.item()
			_, predicted = torch.max(output.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).sum()
			acc = 100. * correct / total
			# sys.stdout.write('\r')
			# sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
			#                  % (epoch, num_epochs, i + 1,
			#                     (len(train_data) // batch_size) + 1, loss.item(), acc))
			# sys.stdout.flush()
		print(f" =>Epoch[{epoch}] acc:{acc} lr:{scheduler.get_lr()[0]}  Time: {(time.time() - end_epoch_time) / 60:8.2f}m")
		if epoch % 10 == 0 and test_data is not None:
			fewshot_acc_1_test, fewshot_acc_5_test = accuracy(model, test_data, batch_size=batch_size, aug_param=aug_param)
			print(f"\n=> Mid accuracy@1: {fewshot_acc_1_test} | accuracy@5: {fewshot_acc_5_test}\n")
			model.train()
		# if epoch % 5 == 0:
		# 	checkpoint(acc, epoch, model, seed, name)
		scheduler.step(epoch)
		# save_checkpoint(f'baseline/{name}', epoch, model)


# def train_linear_classifier_2_datasets(model, criterion, optimizer, train_data1, train_data2, test_data, num_epochs,
#                                        resolution=84, offset_idx=0, offset_label=0, batch_size=128):
# 	model.train()
# 	dataloaders_large = get_loader_with_idx(train_data1, batch_size, image_size=resolution, num_workers=8,
# 	                                        augment=False)
# 	dataloaders_small = get_loader_with_idx(train_data2, batch_size, image_size=resolution, num_workers=8,
# 	                                        augment=False,
# 	                                        offset_idx=offset_idx, offset_label=offset_label)
#
# 	# train_data_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True, num_workers=2, pin_memory=False)
#
# 	for epoch in range(num_epochs):
# 		for i, batch in enumerate(dataloaders_large):
# 			# dataloader_iterator = iter(dataloaders_small)
# 			# idx = batch[0]
# 			imgs = maybe_cuda(batch[1])
# 			targets1 = maybe_cuda(batch[2])
# 			optimizer.zero_grad()
# 			if i <= len(train_data2):
# 				data2 = next(dataloader_iterator)
# 			else:
# 				dataloader_iterator = iter(dataloaders_small)
# 				data2 = next(dataloader_iterator)
# 			imgs2 = maybe_cuda(data2[1])
# 			targets2 = maybe_cuda(data2[2])
# 			targets = torch.cat([targets1, targets2])
# 			output = model(torch.cat([imgs, imgs2]).squeeze())
# 			# output = model(imgs).squeeze()
# 			# _, preds = torch.max(output.data, 1)
# 			loss = criterion(output, targets.view(-1, 1).squeeze()).mean()
# 			loss.backward()
# 			optimizer.step()
# 		print(f"Epoch:{epoch}, loss:{loss}")
# 		acc_1_valid, acc_5_valid = accuracy(model, train_data1)
# 		fewshot_acc_1_valid, fewshot_acc_5_valid = accuracy(model, train_data2)
# 		test_acc_1_valid, test_acc_5_valid = accuracy(model, test_data)
# 		print('acc_1_valid accuracy@1', acc_1_valid)
# 		print('acc_5_valid accuracy@5', acc_5_valid)
# 		print('fewshot_acc_1_valid accuracy@1', fewshot_acc_1_valid)
# 		print('fewshot_acc_5_valid accuracy@1', fewshot_acc_5_valid)
# 		print('test_acc_1_valid accuracy@1', test_acc_1_valid)
# 		print('test_acc_5_valid accuracy@1', test_acc_5_valid)


# for i, batch in enumerate(dataloaders_small):
# 	idx = batch[0]
# 	imgs = maybe_cuda(batch[1])
# 	targets1 = maybe_cuda(batch[2])
# 	optimizer.zero_grad()
# 	output = model(imgs).squeeze()
# 	loss = criterion(output, targets1.view(-1, 1).squeeze())#.mean()
# 	loss.backward()
# 	optimizer.step()
# 	if loss < best_loss:
# 		best_loss = loss
# print(f"Epoch:{epoch}, loss:{loss}")
# acc_1_valid, acc_5_valid = accuracy(model, train_data1)
# fewshot_acc_1_valid, fewshot_acc_5_valid = accuracy(model, train_data2)
# test_acc_1_valid, test_acc_5_valid = accuracy(model, test_data)
# print('acc_1_valid accuracy@1', acc_1_valid)
# print('acc_5_valid accuracy@5', acc_5_valid)
# print('fewshot_acc_1_valid accuracy@1', fewshot_acc_1_valid)
# print('fewshot_acc_5_valid accuracy@1', fewshot_acc_5_valid)
# print('test_acc_1_valid accuracy@1', test_acc_1_valid)
# print('test_acc_5_valid accuracy@1', test_acc_5_valid)


# @easyargs
# def run_eval(seed=0, debug=False):
# 	image_path_train = '/cs/labs/daphna/daphna/data/miniimagenet/train'
# 	image_path_test = '/cs/labs/daphna/daphna/data/miniimagenet/test'
# 	PATH = "/cs/labs/daphna/idan.azuri/myglo/glo/"
# 	if debug:
# 		image_path_test = image_path_train = '/Users/idan.a/repos/data/miniimagenet/test'
# 		PATH = "/Users/idan.a/repos/myglo/glo/"
# 	train_data = DatasetFolderFromList(image_path_train)
# 	offset_idx = max(train_data.path2idx.values()) + 1
# 	offset_labels = max(train_data.targets) + 1
# 	fewshot_train_data = DatasetFolderFromList(image_path_test, class_cap=500, offset_idx=offset_idx)
# 	updated_offset_idx = max(fewshot_train_data.path2idx.values()) + 1
# 	fewshot_test_data = DatasetFolderFromList(image_path_test, offset_idx=updated_offset_idx,
# 	                                          ignore_samples_by_path=fewshot_train_data.path2idx)
# 	random.seed(seed)
# 	torch.manual_seed(seed)
# 	np.random.seed(seed)
# 	print(f"train_data: {len(train_data)}")
# 	print(f"fewshot_train_data: {len(fewshot_train_data)}")
# 	print(f"test data: {len(fewshot_test_data)}")
# 	print(f"train_data.classes: {len(train_data.classes)}")
# 	print(f"fewshot_train_data.classes: {len(fewshot_train_data.classes)}")
# 	total_classes = len(train_data.classes) + len(fewshot_train_data.classes)
# 	print(f"total classes: {total_classes}")
#
# 	# data_train, valid_data_test = train_valid_split(train_data, split_fold=10, random_seed=seed)
# 	# fewshot_data_for_train, fewshot_data_for_test = train_valid_split(fewshot_train_data, split_fold=10, random_seed=seed)
#
# 	# classifier = cnn(num_classes=total_classes, im_size=resolution)  # Pixel space
# 	# classifier = models.resnet152(pretrained=False, num_classes=100)
# 	# classifier = WideResNet(depth=28, num_classes=100, widen_factor=10, dropRate=0.3)
# 	classifier = cnn(num_classes=100, im_size=84)  # Pixel space
# 	criterion = nn.CrossEntropyLoss()
# 	num_gpus = torch.cuda.device_count()
# 	if num_gpus > 1:
# 		print(f"=> Using {num_gpus} GPUs")
# 		classifier = nn.DataParallel(classifier.cuda(), device_ids=list(range(num_gpus)))
# 		# criterion = nn.DataParallel(classifier.cuda(), device_ids=list(range(num_gpus)))
# 		criterion = criterion.cuda()
# 	else:
# 		classifier = maybe_cuda(classifier)
# 		criterion = maybe_cuda(criterion)
# 	optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
# 	train_linear_classifier_2_datasets(classifier, criterion, optimizer, train_data, fewshot_train_data,
# 	                                   test_data=fewshot_test_data, num_epochs=30, offset_idx=offset_idx,
# 	                                   offset_label=offset_labels)
# 	# train_linear_classifier(classifier,criterion, optimizer, fewshot_train_data, num_epochs=30, offset_idx=offset_idx, offset_label=offset_labels)
# 	# train_linear_classifier(classifier,criterion, optimizer, train_data, num_epochs=20)
#
# 	acc_1_valid, acc_5_valid = accuracy(classifier, train_data)
# 	fewshot_acc_1_valid, fewshot_acc_5_valid = accuracy(classifier, fewshot_train_data)
# 	fewshot_acc_1_test, fewshot_acc_5_test = accuracy(classifier, fewshot_test_data)
#
# 	print('acc_1_valid accuracy@1', acc_1_valid)
# 	print('acc_5_valid accuracy@5', acc_5_valid)
# 	print('fewshot_acc_1_valid accuracy@1', fewshot_acc_1_valid)
# 	print('fewshot_acc_5_valid accuracy@1', fewshot_acc_5_valid)
# 	print('fewshot_acc_1_test accuracy@1', fewshot_acc_1_test)
# 	print('fewshot_acc_5_test accuracy@5', fewshot_acc_5_test)
# 	# name = rn.split("_")[1:-5]
# 	# name='_'.join(name[:-1])
# 	# training data
#
# 	w = csv.writer(open(f"CNN_test_acc1.csv", "w+"))
# 	w.writerow(fewshot_acc_1_test)
#
# 	w = csv.writer(open(f"CNN_test_acc5.csv", "w+"))
# 	w.writerow(fewshot_acc_5_test)


@easyargs
def run_eval_generic(epochs=200, d="", fewshot=False, augment=False, autoaugment=False, data="", shot=None,
                     resume=False, pretrained=False,batch_size=128):
	global classifier
	dir = "baselines_aug"
	if not os.path.isdir(dir):
		os.mkdir(dir)
	name = f"{data}_{d}_baseline_"
	if not shot is None:
		name = name + f"shot{shot}"
	if not autoaugment is None:
		name = name + "_aug"

	logger = Logger(os.path.join(f"{dir}/", f'{d}_log.txt'), title=name)
	logger.set_names(["valid_acc@1", "valid_acc@5", "test_acc@1", "test_acc@5"])
	data_dir = '../../data'
	cifar_dir_small="../../data/cifar-100"
	aug_param = aug_param_test=None

	if data == "cifar":
		# batch_size = 128
		lr = 0.1
		classes = 100
		WD = 5e-4
		if not fewshot:
			train_data = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
			transductive_train_data = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
			test_data = transductive_train_data
		else:
			print("=> Fewshot")
			train_data, transductive_train_data = get_cifar100_small(cifar_dir_small, shot)
			test_data = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
		print(f"train_data size:{len(train_data)}")
		print(f"test_data size:{len(test_data)}")
	if data == "cub":
		print("CUB-200")
		classes = 200
		# batch_size = 16
		lr = 0.001
		WD = 1e-5
		aug_param = aug_param_test=get_cub_param()
		train_data = Cub2011(root=f"../../data/{data}", train=True)
		test_data = Cub2011(root=f"../../data/{data}", train=False)
		print(f"train_data size:{len(train_data)},{train_data.data.shape}")
		print(f"test_data size:{len(test_data)},{test_data.data.shape}")
	if data == "stl":
		print("STL-10")
		classes = 10
		WD=4e-4
		# batch_size = 128
		lr=2e-3
		aug_param = get_train_n_unlabled_stl_param()
		aug_param_test = get_test_stl_param()
		train_data = torchvision.datasets.STL10(root=f"../../data/{data}", split='train', download=True)
		# transductive_train_data = torchvision.datasets.STL10(root=f"../../data/{data}", split='unlabeled', download=True)
		test_data = torchvision.datasets.STL10(root=f"../../data/{data}", split='test', download=True)
		print(f"train_data size:{len(train_data)},{train_data.data.shape}")
		print(f"test_data size:{len(test_data)},{test_data.data.shape}")
	scores_test_acc1_fewshot = dict()
	scores_test_acc5_fewshot = dict()
	for seed in [1, 17, 32]:
		# offset_idx = len(train_data)
		print(f"{data} num classes: {classes}")
		random.seed(seed)
		torch.manual_seed(seed)
		np.random.seed(seed)
		start_epoch = 0
		if resume:
			# Load checkpoint.
			print('==> Resuming from checkpoint..')
			assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
			checkpoint = torch.load(f'./checkpoint/{name}_{seed}_ckpt.t7')
			classifier = checkpoint['net']
			best_acc = checkpoint['acc']
			start_epoch = checkpoint['epoch'] + 1
			rng_state = checkpoint['rng_state']
			torch.set_rng_state(rng_state)
			print(f"=> Model loaded start_epoch{start_epoch}, acc={best_acc}")

		else:
			classifier = get_classifier(classes, d, pretrained)
		criterion = nn.CrossEntropyLoss().cuda()
		num_gpus = torch.cuda.device_count()
		if num_gpus > 1:
			classifier = nn.DataParallel(classifier).cuda()
		else:
			classifier = classifier.cuda()
		optimizer = optim.SGD(classifier.parameters(), lr, momentum=0.9, weight_decay=WD, nesterov=True)
		cudnn.benchmark = True
		print(' => Total params: %.2fM' % (sum(p.numel() for p in classifier.parameters()) / 1000000.0))
		print(f"=> {d}  Training model")
		print(f"=> Training Epochs = {str(epochs)}")
		print(f"=> Initial Learning Rate = {str(lr)}")

		# criterion = maybe_cuda(criterion)
		# optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
		train_linear_classifier(classifier, criterion, optimizer, seed=seed, name=name, train_data=train_data,
		                        batch_size=batch_size,
		                        num_epochs=epochs,
		                        fewshot=fewshot,
		                        augment=augment, autoaugment=autoaugment, aug_param=aug_param, test_data=test_data,
		                        start_epoch=start_epoch)

		acc_1_valid, acc_5_valid = accuracy(classifier, train_data, batch_size=batch_size, aug_param=aug_param)
		fewshot_acc_1_test, fewshot_acc_5_test = accuracy(classifier, test_data, batch_size=batch_size, aug_param=aug_param_test)

		print('fewshot_acc_1_valid accuracy@1', acc_1_valid)
		print('fewshot_acc_5_valid accuracy@1', acc_5_valid)

		print('fewshot_acc_1_test accuracy@1', fewshot_acc_1_test)
		print('fewshot_acc_5_test accuracy@5', fewshot_acc_5_test)
		logger.append([acc_1_valid, acc_5_valid, fewshot_acc_1_test, fewshot_acc_5_test])
		scores_test_acc1_fewshot[seed] = fewshot_acc_1_test
		scores_test_acc5_fewshot[seed] = fewshot_acc_5_test

	w = csv.writer(open(f"baseline_shot_{shot}_acc1.csv", "w+"))
	for key, val in scores_test_acc1_fewshot.items():
		w.writerow([key, val])
	w = csv.writer(open(f"baseline_shot_{shot}_acc5.csv", "w+"))
	for key, val in scores_test_acc5_fewshot.items():
		w.writerow([key, val])
	logger.close()


def checkpoint(acc, epoch, net, seed, name):
	# Save checkpoint.
	print('Saving..')
	state = {
		'net': net,
		'acc': acc,
		'epoch': epoch,
		'rng_state': torch.get_rng_state()
	}
	if not os.path.isdir('checkpoint'):
		os.mkdir('checkpoint')
	torch.save(state, f'./checkpoint/{name}_{seed}_ckpt.t7')


if __name__ == '__main__':
	run_eval_generic()
	print("Done!")
