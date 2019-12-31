#!/usr/bin/env python3
import csv
import glob
import os
import random
import time
import easyargs
import matplotlib
import torch
from torch import optim
from torch.backends import cudnn
from torch.optim import lr_scheduler

from utils import Logger

matplotlib.use('Agg')
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
os.sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# from cub2011 import Cub2011
from glo.interpolate import slerp_torch
from glo.utils import *
from glo.model import _netZ, _netG, Classifier, DCGAN_G, DCGAN_G_small


# dim = 512
code_size = 100


class LinearSVM(nn.Module):
	"""Support Vector Machine"""

	def __init__(self, num_class, dim):
		super(LinearSVM, self).__init__()
		self.fc = nn.Linear(dim, num_class)

	# self.sgi = nn.Sigmoid()

	def forward(self, x):
		h = self.fc(x)
		return h  # , torch.argmax(h,1)


# normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])  # CIFAR100
#
# transform_list = normalize


def accuracy(model, test_data, batch_size=128, topk=(1, 5), aug_param=None):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	model.eval()
	if aug_param is None:
		aug_param = {'std': None, 'mean': None, 'rand_crop': 32, 'image_size': 32}
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	with torch.no_grad():
		test_loader = get_loader_with_idx(test_data, batch_size=batch_size, **aug_param, num_workers=8,
		                                  shuffle=False, eval=True)
		for i, batch in enumerate(test_loader):
			imgs = maybe_cuda(batch[1])
			targets = maybe_cuda(batch[2])
			output = maybe_cuda(model(imgs))
			maxk = max(topk)
			batch_size = targets.size(0)
			# target = validate_loader_consistency(batch_size, idx, target, test_data)
			_, pred = output.topk(maxk, 1, True, True)
			pred = pred.t()
			correct = pred.eq(targets.view(1, -1).expand_as(pred))
			res = []
			for k in topk:
				correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
				res.append(correct_k.mul_(100.0 / batch_size))
			top1.update(res[0].item())
			top5.update(res[1].item())
	return top1.avg, top5.avg


def generic_train_classifier(model, optimizer, train_data, num_epochs, is_inter, criterion, offset_idx=0,
                             offset_label=0, batch_size=128, fewshot=False,
                             augment=False, autoaugment=False, test_data=None, loss_method="ce", n_classes=100,
                             aug_param=None, aug_param_test=None):
	model = model.cuda()
	model.train()
	sampler = None
	best_acc = 0
	milestones = [60, 120, 160]
	if fewshot:
		num_epoch_repetition = fewshot_setting(train_data)
		milestones = list(map(lambda x: num_epoch_repetition * x, milestones))
		num_epochs *= num_epoch_repetition
	train_data_loader = get_loader_with_idx(train_data, batch_size,
	                                        augment=augment,
	                                        offset_idx=offset_idx, offset_label=offset_label, sampler=sampler,
	                                        autoaugment=autoaugment, **aug_param)
	print(f" train_len:{len(train_data)}")
	correct = 0
	total = 0
	if aug_param['image_size'] == 256:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
	else:
		scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
	for epoch in range(num_epochs):
		netG.eval()
		netZ.eval()
		model.train()
		end_epoch_time = time.time()

		for i, (idx, inputs, _) in enumerate(train_data_loader):
			inputs = inputs.cuda()
			optimizer.zero_grad()
			targets = validate_loader_consistency(netZ, idx)
			output = calc_output(idx, is_inter, model, targets, inputs, aug_param)
			targets = torch.tensor(targets).long().cuda()
			if loss_method == "cosine":
				y_target = torch.ones(len(idx)).cuda()
				onehot_labels = one_hot(targets, n_classes)
				loss = criterion(output, onehot_labels.cuda(), y_target)
			else:
				loss = criterion(output, targets)
			loss.backward()
			optimizer.step()
			_, predicted = torch.max(output.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets.data).sum()
		# if i % 100 ==0 :
		# sys.stdout.write('\r')
		# sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
		#                  % (epoch, num_epochs, i + 1,
		#                     (len(train_data) // batch_size) + 1, loss.item(), 100. * correct / total))
		# sys.stdout.flush()
		print(
			f"Epoch [{epoch}/{num_epochs}] Acc:{100. * correct / total:2.2f}% loss:{loss.item():4.2f} lr:{scheduler.get_lr()[0]}  Time: {(time.time() - end_epoch_time) / 60:4.2f}m")
		scheduler.step(epoch)
		if epoch % 20 == 0 and test_data is not None:
			trans_acc_1, trans_acc_5 = accuracy(model, test_data, batch_size=batch_size, aug_param=aug_param_test)
			if trans_acc_1 > best_acc:
				best_acc = trans_acc_1
			print(f"\n=> Epoch[{epoch}] mid accuracy@1: {trans_acc_1} | accuracy@5: {trans_acc_5}\n")
	print(f"Best Acc 1 = {best_acc}")


def fewshot_setting(train_data):
	print("=> fewshot")
	data_targets = get_labels(train_data)
	class_sample_count = np.unique(data_targets, return_counts=True)[1]
	## uncommnent if class imbalanced
	# weight = 1. / class_sample_count
	# samples_weight = torch.from_numpy((weight[data_targets]))
	# print(samples_weight.min())
	# print(samples_weight.max())
	# sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
	num_epoch_repetition = 500 // class_sample_count[0]
	print(f"num_epoch_repetition: {num_epoch_repetition}")
	return num_epoch_repetition


def calc_output(idx, is_inter, model, targets, inputs, aug_param):
	# Zs_real_numpy = netZ.emb.weight.data.cpu().numpy()
	normalize = transforms.Normalize(mean=aug_param['mean'], std=aug_param['std'])
	Zs_real = netZ.emb.weight.data
	if is_inter:
		# z = Zs_real_numpy[idx]
		z = Zs_real[idx]
		# z = z.detach().cpu().numpy()
		# z_knn_dist, z_knn = neigh.kneighbors(z, return_distance=True)
		# rnd_int = random.randint(1, 3)
		# z_knn = z_knn[:, 1:3]  # ignore self dist
		z_knn_idx = np.array([random.sample(netZ.label2idx[x], 1) for x in targets])

		# z_knn_idx = z_knn_idx[:, 0]# * 0.7 + z_knn[:, 1] * 0.3

		ratios = list(np.linspace(0.1, 0.4, 5))
		rnd_ratio = random.sample(ratios, 1)[0]
		# z = torch.from_numpy(z).float().cuda()
		z = z.float().cuda()
		z_knn = Zs_real[z_knn_idx[:, 0]]  # * 0.9 + Zs_real[z_knn_idx[:, 1]] * 0.1
		# eps=maybe_cuda(torch.FloatTensor(len(idx), dim).normal_(0, 0.01))
		inter_z_slerp = slerp_torch(rnd_ratio, z.unsqueeze(0), z_knn.cuda().float().unsqueeze(0))
		# inter_z = torch.lerp(z, z_knn.float(), rnd_ratio)
		# inter_z_slerp = interpolate_points(z, z_knn.float(), 10, print_mode=False, slerp=True)
		# inter_z = interpolate_points(z,z_knn.float(),10,print_mode=False,slerp=False)
		# targets = torch.tensor(targets).long().cuda()
		# targets = targets.repeat(10)

		code = get_code(idx)
		if random.random() > 0.5:
			generated_img = netG(inter_z_slerp.squeeze().cuda(), code)
			# generated_img_2 = netG(inter_z.squeeze().cuda(),code)
			# code = torch.cuda.FloatTensor(len(idx), code_size).normal_(0, 0.15)
			# recon = netG(z.squeeze(),code)
			## for all list of transformations
			# imgs = torch.stack([transform_list(transforms.ToPILImage()(img)) for img in generated_img.detach().cpu()])
			imgs = torch.stack([normalize((img)) for img in generated_img.clone()])
		# imgs2 = torch.stack([normalize((img)) for img in generated_img_2.clone()])
		# recon_imgs = torch.stack([normalize((img)) for img in recon.clone()])
		# save_image_grid(imgs.data, f'runs/extrapol_z_slerp.png', ngrid=10)
		# save_image_grid(imgs2.data, f'runs/extrapol_z_lerp.png', ngrid=10)
		# save_image_grid(inputs.data, f'runs/in.png', ngrid=10)
		# save_image_grid(recon_imgs.data, f'runs/recon.png', ngrid=10)
		# print("SAVED")
		# exit()
		else:
			imgs = inputs  # the input is augmented
		output = model(imgs.cuda()).squeeze()  # .cpu().numpy()
	# save_image_grid(imgs.data, f'runs/inter_norm_slerp.png', ngrid=inter_steps)
	# targets = targets.repeat((1, inter_steps))

	else:
		code = get_code(idx)
		if random.random() > 0.5:
			generated_img = netG(Zs_real[idx].cuda(), code)
			imgs = torch.stack([normalize((img)) for img in generated_img.clone()])
		else:
			imgs = inputs
		output = model(imgs.cuda())
	return output


def get_code(idx):
	code = torch.cuda.FloatTensor(len(idx), code_size).normal_(0, 0.15)
	# normed_code = code.norm(2, 1).detach().unsqueeze(1).expand_as(code)
	# code = code.div(normed_code)
	return code


# def accuracy(model, test_data, offset_idx, offset_label, topk=(1, 5), is_pixel=True, aug_param=None, batch_size=32):
# 	"""Computes the accuracy over the k top predictions for the specified values of k"""
# 	print(f"is_pixel={is_pixel}")
# 	model = model.cuda()
# 	top1 = AverageMeter('Acc@1', ':6.2f')
# 	top5 = AverageMeter('Acc@5', ':6.2f')
# 	with torch.no_grad():
# 		test_loader = get_loader_with_idx(test_data, batch_size=batch_size, shuffle=False,
# 		                                  offset_idx=offset_idx, offset_label=offset_label, eval=True, **aug_param)
# 		for i, batch in enumerate(test_loader):
# 			# idx = batch[0]
# 			imgs = maybe_cuda(batch[1])
# 			target = maybe_cuda(batch[2])
# 			output = model(imgs)
# 			maxk = max(topk)
# 			batch_size = target.size(0)
# 			_, pred = output.topk(maxk, 1, True, True)
# 			pred = pred.t()
# 			correct = pred.eq(target.view(1, -1).expand_as(pred))
#
# 			res = []
# 			for k in topk:
# 				correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
# 				res.append(correct_k.mul_(100.0 / batch_size))
# 			top1.update(res[0].item())
# 			top5.update(res[1].item())
# 	return top1.avg, top5.avg


# @easyargs
# def run_eval(seed=0, is_inter=False, debug=False, resolution=84, is_model_classifier=False, keyword="", is_pixel=True):
# 	global netG, netZ, Zs_real, neigh
#
# 	image_path_train = '/cs/labs/daphna/daphna/data/miniimagenet/train'
# 	image_path_test = '/cs/labs/daphna/daphna/data/miniimagenet/test'
# 	PATH = "/cs/labs/daphna/idan.azuri/myglo/glo/"
# 	if debug:
# 		image_path_test = image_path_train = '/Users/idan.a/repos/data/miniimagenet/test'
# 		PATH = "/Users/idan.a/repos/myglo/glo/"
# 	train_data = DatasetFolderFromList(image_path_train)
# 	offset_idx = max(train_data.path2idx.values()) + 1
# 	offset_labels = max(train_data.targets) + 1
# 	fewshot_data = DatasetFolderFromList(image_path_test, class_cap=500, offset_idx=offset_idx)
# 	updated_offset_idx = max(fewshot_data.path2idx.values()) + 1
# 	test_data_transductive = DatasetFolderFromList(image_path_test, offset_idx=updated_offset_idx,
# 	                                               ignore_samples_by_path=fewshot_data.path2idx)
#
# 	classes_ = len(train_data.classes) + len(fewshot_data.classes)
# 	cnn_ = cnn(num_classes=classes_, im_size=resolution)
# 	linear = Classifier(classes_, dim)
# 	random.seed(0)
# 	torch.manual_seed(0)
# 	np.random.seed(0)
#
# 	# netZ.set_label2idx()
# 	netG = _netG(dim, resolution, 3, classes_)
# 	netG = maybe_cuda(netG)
#
# 	print(len(train_data))
# 	print(len(fewshot_data))
# 	print(len(train_data.classes))
# 	print(len(fewshot_data.classes))
# 	paths = list()
# 	dirs = [d for d in glob.iglob(PATH)]
#
# 	for dir in dirs:
# 		for f in glob.iglob(f"{dir}runs/*{keyword}*log.txt"):
# 			# for f in glob.iglob(f"{dir}runs/*log.txt"):
# 			fname = f.split("/")[-1]
# 			tmp = fname.split("_")
# 			name = '_'.join(tmp[:-1])
# 			print(name, f)
# 			if is_model_classifier:
# 				if "classifier" in name or "cnn" in name or "pixel" in name:
# 					paths.append(name)
# 			else:
# 				paths.append(name)
# 	print(f"Total runs: {len(paths)}\n{paths}")
# 	scores_train_acc1 = dict()
# 	scores_test_acc1 = dict()
# 	scores_train_acc5 = dict()
# 	scores_test_acc5 = dict()
# 	scores_train_acc1_fewshot = dict()
# 	scores_train_acc5_fewshot = dict()
# 	scores_test_acc1_fewshot = dict()
# 	scores_test_acc5_fewshot = dict()
# 	for rn in paths:
# 		print(rn)
# 		if not "tr" in rn:
# 			netZ = _netZ(dim, len(fewshot_data) + len(train_data), len(train_data.classes), None)
# 		else:
# 			netZ = _netZ(dim, len(fewshot_data) + len(train_data) + len(test_data_transductive),
# 			             len(train_data.classes), None)
# 		try:
# 			print(f"=> Loading model from {rn}")
# 			_, netG = load_saved_model(f'runs/nets_{rn}/netG_nag', netG)
# 			epoch, netZ = load_saved_model(f'runs/nets_{rn}/netZ_nag', netZ)
# 			print(len(netZ.emb.weight))
# 			netZ = maybe_cuda(netZ)
# 			netG = maybe_cuda(netG)
# 			if epoch > 0:
# 				print(f"=> Loaded successfully! epoch:{epoch}")
# 			else:
# 				print("=> No checkpoint to resume")
# 		except Exception as e:
# 			print(f"=>Failed resume job!\n {e}")
# 		netG.eval()
# 		netZ.eval()
# 		Zs_real = netZ.emb.weight.data.cpu().numpy()
# 		# if is_inter:
#
# 		# train_data_split, test_data_split = train_valid_split(dataset, split_fold=10, random_seed=seed)
# 		# train_data_for_train, train_data_for_test = train_valid_split(train_data, split_fold=10, random_seed=seed)
# 		# fewshot_data_for_train, fewshot_data_for_test = train_valid_split(fewshot_data, split_fold=10, random_seed=seed)
# 		# print(len(train_data_for_train), len(train_data_for_test))
# 		# print(f"train size:{len(train_data_for_train) + len(fewshot_data_for_train)}, test size:{len(train_data_for_test) + len(fewshot_data_for_test)}")
#
# 		if "pixel" in rn:
# 			is_pixel = True
# 			print("=> CNN")
# 			classifier = cnn_
# 		else:
# 			is_pixel = False
# 			print("=> Linear")
# 			classifier = linear
# 		if is_model_classifier:
# 			epoch, classifier = load_saved_model(f'runs/nets_{rn}/netD_nag', classifier)
# 			classifier = maybe_cuda(classifier)
# 			if epoch < 50:
# 				continue
# 		else:
#
# 			print("=> New training new classifier")
# 			classifier.apply(weights_init)
# 			optimizer = optim.Adam(classifier.parameters(), lr=0.1, weight_decay=5e-4)
# 			print("=> Train classifier")
# 			criterion = nn.CrossEntropyLoss()
# 			num_gpus = torch.cuda.device_count()
# 			if num_gpus > 1:
# 				print(f"=> Using {num_gpus} GPUs")
# 				classifier = nn.DataParallel(classifier.cuda(), device_ids=list(range(num_gpus)))
# 				# criterion = nn.DataParallel(classifier.cuda(), device_ids=list(range(num_gpus)))
# 				criterion = criterion.cuda()
# 			else:
# 				classifier = maybe_cuda(classifier)
# 				criterion = maybe_cuda(criterion)
# 			generic_train_classifier(classifier, optimizer, train_data, criterion=criterion, resolution=resolution,
# 			                         is_inter=is_inter, num_epochs=20, is_pixel=is_pixel)
# 			generic_train_classifier(classifier, optimizer, fewshot_data, criterion=criterion, resolution=resolution,
# 			                         is_inter=is_inter, num_epochs=20, offset_idx=offset_idx,
# 			                         offset_label=offset_labels,
# 			                         is_pixel=is_pixel)
#
# 		fewshot_acc_1_test, fewshot_acc_5_test = accuracy(classifier, test_data_transductive, resolution=resolution,
# 		                                                  is_pixel=is_pixel, offset_idx=updated_offset_idx,
# 		                                                  offset_label=offset_labels)
# 		print('test_data_transductive accuracy@1', fewshot_acc_1_test)
# 		print('test_data_transductive accuracy@5', fewshot_acc_5_test)
# 		train_acc_1_train, train_acc_5_train = accuracy(classifier, train_data, resolution=resolution,
# 		                                                is_pixel=is_pixel, offset_idx=0,
# 		                                                offset_label=0)
# 		fewshot_data_1, fewshot_data_5 = accuracy(classifier, fewshot_data, resolution=resolution, is_pixel=is_pixel,
# 		                                          offset_idx=offset_idx,
# 		                                          offset_label=offset_labels)
#
# 		print('train_acc_1_train accuracy@1', train_acc_1_train)
# 		print('train_acc_5_train accuracy@5', train_acc_5_train)
# 		print('fewshot_data_1 accuracy@1', fewshot_data_1)
# 		print('fewshot_data_5 accuracy@5', fewshot_data_5)
# 		# print('fewshot_acc_1_train accuracy@1', fewshot_acc_1_train)
# 		# print('fewshot_acc_5_train accuracy@5', fewshot_acc_5_train)
# 		# name = rn.split("_")[1:-5]
# 		# name='_'.join(name[:-1])
# 		# training data
# 		# scores_train_acc1[rn] = train_acc_1_train
# 		# scores_test_acc1[rn] = train_acc_1_test
# 		# scores_train_acc5[rn] = train_acc_5_train
# 		# scores_test_acc5[rn] = train_acc_5_test
# 		# # few shot data
# 		# scores_train_acc1_fewshot[rn] = fewshot_acc_1_train
# 		scores_test_acc1_fewshot[rn] = fewshot_acc_1_test
# 		# scores_train_acc5_fewshot[rn] = fewshot_acc_5_train
# 		scores_test_acc5_fewshot[rn] = fewshot_acc_5_test
# 	info = f'_seed_{seed}'
# 	if is_inter:
# 		info = info + '_inter'
# 	if is_pixel:
# 		info = info + '_pixel'
# 	if is_model_classifier:
# 		info = info + '_classifier'
#
# 	# w = csv.writer(open(f"{keyword}_scores_train_acc1{info}.csv", "w+"))
# 	# for key, val in scores_train_acc1.items():
# 	# 	w.writerow([key, val])
# 	# w = csv.writer(open(f"{keyword}_scores_test_acc1{info}.csv", "w+"))
# 	# for key, val in scores_test_acc1.items():
# 	# 	w.writerow([key, val])
# 	# w = csv.writer(open(f"{keyword}_scores_train_acc5{info}.csv", "w+"))
# 	# for key, val in scores_train_acc5.items():
# 	# 	w.writerow([key, val])
# 	# w = csv.writer(open(f"{keyword}_scores_test_acc5{info}.csv", "w+"))
# 	# for key, val in scores_test_acc5.items():
# 	# 	w.writerow([key, val])
# 	#
# 	# w = csv.writer(open(f"{keyword}_scores_train_acc1_fewshot{info}.csv", "w+"))
# 	# for key, val in scores_train_acc1_fewshot.items():
# 	# 	w.writerow([key, val])
# 	w = csv.writer(open(f"{keyword}_scores_test_acc1_fewshot{info}.csv", "w+"))
# 	for key, val in scores_test_acc1_fewshot.items():
# 		w.writerow([key, val])
# 	w = csv.writer(open(f"{keyword}_scores_test_acc5_fewshot{info}.csv", "w+"))
# 	for key, val in scores_test_acc5_fewshot.items():
# 		w.writerow([key, val])


# w = csv.writer(open(f"{keyword}_scores_train_acc5_fewshot{info}.csv", "w+"))
# for key, val in scores_train_acc5_fewshot.items():
# 	w.writerow([key, val])


@easyargs
def run_eval_cifar(is_inter=False, debug=False, keyword="", epochs=200, d="", fewshot=False, augment=False, shot=50,
                   autoaugment=False,
                   loss_method="ce", data="cifar", pretrained=False, dim=512):
	global netG, netZ, Zs_real, neigh, aug_param, criterion, aug_param_test
	title = f"{d}_{keyword}"
	if fewshot:
		title = title + "_fewshot"

	if is_inter:
		title = title + "_inter"
	dir = "eval"
	if not os.path.isdir(dir):
		os.mkdir(dir)
	logger = Logger(os.path.join(f'{dir}/', f'{title}_log.txt'), title=title)
	logger.set_names(["valid_acc@1", "valid_acc@5", "test_acc@1", "test_acc@5"])
	PATH = "/cs/labs/daphna/idan.azuri/myglo/glo/"
	if debug:
		PATH = "/Users/idan.a/repos/myglo/glo/"
	data_dir = '../../data'
	cifar_dir_small = "../../data/cifar-100"
	if data == "cifar":
		classes = 100
		lr = 0.1
		batch_size = 128
		WD = 5e-4
		aug_param = aug_param_test = get_cifar_param()
		if not fewshot:
			train_data = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
			transductive_train_data = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
			test_data = transductive_train_data
		else:
			print("=> Fewshot")
			# train_data, transductive_train_data = get_cifar100_small(cifar_dir_small, shot)
			test_data = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)
	# if data == "cub":
	# 	classes = 200
	# 	batch_size = 16
	# 	lr = 0.001
	# 	WD = 1e-5
	# 	aug_param = aug_param_test = get_cub_param()
	# 	train_data = Cub2011(root=f"../../data/{data}", train=True)
	# 	test_data = Cub2011(root=f"../../data/{data}", train=False)
	# 	transductive_train_data = test_data
	if data == "stl":
		print("STL-10")
		classes = 10
		WD = 4e-4
		# batch_size = 128
		lr = 2e-3
		aug_param = get_train_n_unlabled_stl_param()
		aug_param_test = get_test_stl_param()
		train_data = torchvision.datasets.STL10(root=f"../../data/{data}", split='train', download=True)
		transductive_train_data = test_data = torchvision.datasets.STL10(root=f"../../data/{data}", split='unlabeled',
		                                                                 download=True)
	train_data_size = len(train_data)
	print(f"train_data size:{train_data_size}")
	print(f"test_data size:{len(test_data)}")
	print(f"transductive data size:{len(transductive_train_data)}")

	linear = Classifier(classes, dim)

	# netZ.set_label2idx()
	noise_projection = keyword.__contains__("proj")
	print(f" noise_projection={noise_projection}")
	# netG = _netG(dim, aug_param['rand_crop'], 3, noise_projection)
	if data == 'cub':
		netG = DCGAN_G(dim, aug_param['rand_crop'], 3, noise_projection)
	elif data == 'stl':
		netG = DCGAN_G_small(dim, aug_param['rand_crop'], 3, noise_projection)
	elif data == 'cifar':
		netG = _netG(dim, aug_param['rand_crop'], 3, noise_projection)
	paths = list()
	dirs = [d for d in glob.iglob(PATH)]

	for dir in dirs:
		for f in glob.iglob(f"{dir}runs/{keyword}*log.txt"):
			# for f in glob.iglob(f"{dir}runs/*log.txt"):
			fname = f.split("/")[-1]
			tmp = fname.split("_")
			name = '_'.join(tmp[:-1])
			# if is_model_classifier:
			# 	if "classifier" in name or "cnn" in name:
			paths.append(name)

	# else:
	# 	paths.append(name)
	scores_test_acc1_fewshot = dict()
	scores_test_acc5_fewshot = dict()
	for seed in [8, 22, 31]:
		set_seed(seed)
		print(f"=> Total runs: {len(paths)}\n{paths}")
		for rn in paths:
			classifier = get_classifier(classes, d, pretrained)
			if "tr_" in rn:
				print("=> Transductive mode")
				netZ = _netZ(dim, train_data_size + len(transductive_train_data), classes, None)
			else:
				print("=> No Transductive")
				netZ = _netZ(dim, train_data_size, classes, None)
			try:
				print(f"=> Loading model from {rn}")
				_, netG = load_saved_model(f'runs/nets_{rn}/netG_nag', netG)
				epoch, netZ = load_saved_model(f'runs/nets_{rn}/netZ_nag', netZ)
				netZ = netZ.cuda()
				netG = netG.cuda()
				print(f"=> Embedding size = {len(netZ.emb.weight)}")

				if epoch > 0:
					print(f"=> Loaded successfully! epoch:{epoch}")
				else:
					print("=> No checkpoint to resume")
			except Exception as e:
				print(f"=> Failed resume job!\n {e}")
			Zs_real = netZ.emb.weight.data.detach().cpu().numpy()
			optimizer = optim.SGD(classifier.parameters(), lr, momentum=0.9, weight_decay=WD, nesterov=True)
			print("=> Train new classifier")
			if loss_method == "cosine":
				criterion = nn.CosineEmbeddingLoss().cuda()
			elif loss_method == "ce":
				criterion = nn.CrossEntropyLoss().cuda()
			num_gpus = torch.cuda.device_count()
			if num_gpus > 1:
				print(f"=> Using {num_gpus} GPUs")
				classifier = nn.DataParallel(classifier).cuda()
				cudnn.benchmark = True
			else:
				classifier = maybe_cuda(classifier)

			print(' => Total params: %.2fM' % (sum(p.numel() for p in classifier.parameters()) / 1000000.0))
			print(f"=> {d}  Training model")
			print(f"=> Training Epochs = {str(epochs)}")
			print(f"=> Initial Learning Rate = {str(lr)}")
			generic_train_classifier(classifier, optimizer, train_data, criterion=criterion, batch_size=batch_size,
			                         is_inter=is_inter, num_epochs=epochs, augment=augment,
			                         fewshot=fewshot, autoaugment=autoaugment, test_data=test_data,
			                         loss_method=loss_method, n_classes=classes, aug_param=aug_param,
			                         aug_param_test=aug_param_test)

			print("=> Done training classifier")

			valid_acc_1, valid_acc_5 = accuracy(classifier, train_data, batch_size=batch_size, aug_param=aug_param_test)
			test_acc_1, test_acc_5 = accuracy(classifier, test_data, batch_size=batch_size, aug_param=aug_param_test)

			# print('fewshot_acc_5_train accuracy@5', fewshot_acc_5_train)
			print('train_acc accuracy@1', valid_acc_1)
			print('train_acc accuracy@5', valid_acc_5)
			print('test accuracy@1', test_acc_1)
			print('test accuracy@5', test_acc_5)
			scores_test_acc1_fewshot[seed] = test_acc_1
			scores_test_acc5_fewshot[seed] = test_acc_5
			logger.append([valid_acc_1, valid_acc_5, test_acc_1, test_acc_5])
			w = csv.writer(open(f"eval_{rn}_shot_{shot}_acc1.csv", "w+"))
			for key, val in scores_test_acc1_fewshot.items():
				w.writerow([rn, key, val])
			w = csv.writer(open(f"eval_{rn}_shot_{shot}_acc5.csv", "w+"))
			for key, val in scores_test_acc5_fewshot.items():
				w.writerow([rn, key, val])

	logger.close()


def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)


if __name__ == '__main__':
	# run_eval()
	run_eval_cifar()
	print("Done!")
