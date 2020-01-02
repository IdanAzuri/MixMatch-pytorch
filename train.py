from __future__ import print_function

import argparse
import glob
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn.parallel
from progress.bar import Bar

import models.wideresnet as models
import dataset.cifar100 as dataset

from glo.interpolate import slerp_torch
from glo.model import _netG, _netZ
from glo.utils import load_saved_model, get_loader_with_idx, get_cifar_param, save_image_grid, \
	validate_loader_consistency
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options

# Method options
parser.add_argument('--n-labeled', type=int, default=50,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=1024,
                    help='Number of labeled data')
parser.add_argument('--out', default='result',
                    help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
# Glo
parser.add_argument('--keyword', default='', type=str, help='path to glo')
parser.add_argument('--dim', default=512, type=int, metavar='N', help='Z dim')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA

use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
	args.manualSeed = 0  # random.randint(1, 10000)
manualSeed = args.manualSeed


def manual_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	# if you are suing GPU
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	print(f"=> SEED = {seed}")


manual_seed(manualSeed)

best_acc = 0  # best test accuracy
code_size = args.dim
print(f"zdim={code_size}")

def get_code(idx):
	code = torch.cuda.FloatTensor(len(idx), code_size).normal_(0, 0.15)
	# normed_code = code.norm(2, 1).detach().unsqueeze(1).expand_as(code)
	# code = code.div(normed_code)
	return code


class TransformTwice:
	def __init__(self, transform):
		self.transform = transform

	def __call__(self, inp):
		out1 = self.transform(inp)
		out2 = self.transform(inp)
		return out1, out2


def main():
	global netG, netZ, Zs_real, aug_param, criterion, aug_param_test
	global best_acc
	aug_param = aug_param_test = get_cifar_param()
	# Data
	print(f'==> Preparing cifar100')
	classes = 100
	normalize = transforms.Normalize(mean=aug_param['mean'], std=aug_param['std'])
	transform_train = transforms.Compose([
		transforms.RandomCrop(32,padding=4),
		transforms.RandomHorizontalFlip(),
		# dataset.RandomPadandCrop(32),
		# dataset.RandomFlip(),
		# dataset.ToTensor(),
		normalize,
	])
	transform_pil = transforms.Compose([
		transforms.ToPILImage(),
		transforms.RandomCrop(32,padding=4),
		transforms.RandomHorizontalFlip(),
		# dataset.RandomPadandCrop(32),
		# dataset.RandomFlip(),
		dataset.ToTensor(),
		normalize,
	])

	transform_val = transforms.Compose([
		dataset.ToTensor(),
		normalize
	])

	train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar100('/cs/dataset/CIFAR/',
	                                                                                 args.n_labeled,
	                                                                                 transform_train=None,
	                                                                                 transform_val=transform_val)
	train_data_size = len(train_labeled_set)
	print(f"train_data size:{train_data_size}")
	print(f"test_data size:{len(test_set)}")
	print(f"train_unlabeled_set data size:{len(train_unlabeled_set)}")
	labeled_trainloader_2 = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
	                                        drop_last=True)
	labeled_trainloader = get_loader_with_idx(train_labeled_set, batch_size=args.batch_size,
	                                          augment=transform_train, drop_last=True, **aug_param)
	# unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
	offset_ = len(train_labeled_set) + len(test_set)
	unlabeled_trainloader = get_loader_with_idx(train_labeled_set, batch_size=args.batch_size,
	                                            augment=TransformTwice(transform_train), drop_last=True,
	                                            offset_idx=offset_, **aug_param)
	val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
	test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
	if not os.path.isdir(args.out):
		mkdir_p(args.out)
	# Model
	print("==> creating WRN-28-2")

	def create_model(ema=False):
		classifier = models.WideResNet(num_classes=100)
		num_gpus = torch.cuda.device_count()
		if num_gpus > 1:
			print(f"=> Using {num_gpus} GPUs")
			classifier = nn.DataParallel(classifier.cuda(), device_ids=list(range(num_gpus))).cuda()
			cudnn.benchmark = True
		else:
			classifier = classifier.cuda()
		if ema:
			for param in classifier.parameters():
				param.detach_()

		return classifier

	classifier = create_model()
	ema_classifier = create_model(ema=True)

	# Loading  pretrained GLO
	keyword = args.keyword
	dim = args.dim
	PATH = "/cs/labs/daphna/idan.azuri/myglo/glo/"
	noise_projection = keyword.__contains__("proj")
	print(f" noise_projection={noise_projection}")
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
	rn = paths[0]
	if "tr_" in rn:
		print("=> Transductive mode")
		netZ = _netZ(dim, train_data_size + len(train_unlabeled_set), classes, None)
	else:
		print("=> No Transductive")
		netZ = _netZ(dim, train_data_size, classes, None)
	try:
		print(f"=> Loading classifier from {rn}")
		_, netG = load_saved_model(f'{dir}runs/nets_{rn}/netG_nag', netG)
		epoch, netZ = load_saved_model(f'{dir}runs/nets_{rn}/netZ_nag', netZ)
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
	# optimizer = optim.SGD(classifier.parameters(), lr, momentum=0.9, weight_decay=WD, nesterov=True)
	# print("=> Train new classifier")
	# if loss_method == "cosine":
	#     criterion = nn.CosineEmbeddingLoss().cuda()
	# elif loss_method == "ce":
	#     criterion = nn.CrossEntropyLoss().cuda()
	print('    Total params: %.2fM' % (sum(p.numel() for p in classifier.parameters()) / 1000000.0))

	train_criterion = SemiLoss()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

	ema_optimizer = WeightEMA(classifier, ema_classifier, alpha=args.ema_decay)
	start_epoch = 0

	# Resume
	title = 'noisy-cifar-100'
	if args.resume:
		# Load checkpoint.
		print('==> Resuming from checkpoint..')
		assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
		args.out = os.path.dirname(args.resume)
		checkpoint = torch.load(args.resume)
		best_acc = checkpoint['best_acc']
		start_epoch = checkpoint['epoch']
		classifier.load_state_dict(checkpoint['state_dict'])
		ema_classifier.load_state_dict(checkpoint['ema_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
	else:
		logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
		logger.set_names(
			['Train Loss', 'Train Loss X', 'Train Loss U', 'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.'])

	step = 0
	test_accs = []
	# Train and val
	for epoch in range(start_epoch, args.epochs):
		print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

		train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, classifier,
		                                               optimizer, ema_optimizer, train_criterion, epoch, use_cuda, transform_pil)
		_, train_acc = validate(labeled_trainloader_2, ema_classifier, criterion, epoch, use_cuda, mode='Train Stats')
		val_loss, val_acc = validate(test_loader, ema_classifier, criterion, epoch, use_cuda, mode='Valid Stats')
		test_loss, test_acc = validate(test_loader, ema_classifier, criterion, epoch, use_cuda, mode='Test Stats ')

		step = args.val_iteration * (epoch + 1)

		# append logger file
		logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_acc, test_loss, test_acc])

		# save classifier
		is_best = val_acc > best_acc
		best_acc = max(val_acc, best_acc)
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': classifier.state_dict(),
			'ema_state_dict': ema_classifier.state_dict(),
			'acc': val_acc,
			'best_acc': best_acc,
			'optimizer': optimizer.state_dict(),
		}, is_best)
		test_accs.append(test_acc)
	logger.close()

	print('Best acc:')
	print(best_acc)

	print('Mean acc:')
	print(np.mean(test_accs[-20:]))


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda,transform):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	losses_x = AverageMeter()
	losses_u = AverageMeter()
	ws = AverageMeter()
	end = time.time()

	bar = Bar('Training', max=args.val_iteration)
	labeled_train_iter = iter(labeled_trainloader)
	unlabeled_train_iter = iter(unlabeled_trainloader)
	Zs_real = netZ.emb.weight.data

	model.train()
	for batch_idx in range(args.val_iteration):
		try:
			idx_x, input_x, targets_x = labeled_train_iter.next()
		except:
			labeled_train_iter = iter(labeled_trainloader)
			idx_x, input_x, targets_x = labeled_train_iter.next()
		try:
			idx_u, (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()
		except:
			unlabeled_train_iter = iter(unlabeled_trainloader)
			idx_u, (inputs_u, inputs_u2), _ = unlabeled_train_iter.next()

		# measure data loading time
		data_time.update(time.time() - end)

		batch_size = idx_x.size(0)

		# Transform label to one-hot
		targets_x = torch.zeros(batch_size, 100).scatter_(1, targets_x.view(-1, 1), 1)

		if use_cuda:
			input_x, targets_x = input_x.cuda(), targets_x.cuda(non_blocking=True)
			inputs_u = inputs_u.cuda()
			inputs_u2 = inputs_u2.cuda()

		with torch.no_grad():
			# compute guessed labels of unlabel samples
			outputs_u = model(inputs_u)
			outputs_u2 = model(inputs_u2)
			p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
			pt = p ** (1 / args.T)
			targets_u = pt / pt.sum(dim=1, keepdim=True)
			targets_u = targets_u.detach()

		# mixup
		ratio = np.random.beta(args.alpha, args.alpha)  # Beta (1, 1) = U (0, 1)
		ratio = max(ratio, 1 - ratio)
		if random.random() > 0.5:  # glo
			all_inputs = torch.cat([idx_x, idx_u, idx_u], dim=0)
			all_inputs_imgs = torch.cat([input_x, inputs_u, inputs_u2], dim=0)
			idx = torch.randperm(all_inputs.size(0))
			idx_a, idx_b = all_inputs, all_inputs[idx]
			input_a, input_b = all_inputs_imgs, all_inputs_imgs[idx]
			input_u1, input_u2 = inputs_u, inputs_u2
			z_a, z_b = Zs_real[idx_a].float().cuda(), Zs_real[idx_b].float().cuda()
			z_u = Zs_real[idx_u].float().cuda()
			# print(f" l:{ratio}")
			inter_z_slerp = torch.lerp(z_a.unsqueeze(0), z_b.unsqueeze(0),ratio)
			code = torch.cuda.FloatTensor(inter_z_slerp.squeeze().size(0), 100).normal_(0, 0.15)
			generated_img = netG(inter_z_slerp.squeeze().cuda(), code)
			#debug
			generated_img_a = netG(z_u.squeeze().cuda(), code)
			save_image_grid(input_u1.data, f'runs/original_u1.png', ngrid=10)
			save_image_grid(input_u2.data, f'runs/original_u2.png', ngrid=10)
			save_image_grid(generated_img_a.data, f'runs/recon_U.png', ngrid=10)
			# generated_imgb = netG(z_b.squeeze().cuda(), code)
			# save_image_grid(generated_img.data, f'runs/generated_img.png', ngrid=10)
			# save_image_grid(generated_imgb.data, f'runs/generated_imgb.png', ngrid=10)
			mixed_input = torch.stack([transform((img)) for img in generated_img.detach().cpu()])  # is it needed?
			save_image_grid(mixed_input.data, f'runs/mixed_transform.png', ngrid=10)
			mixed_input = mixed_input.cuda()
			# save_image_grid(input_x.data, f'runs/originalx.png', ngrid=10)
			# save_image_grid(inputs_u.data, f'runs/originalu.png', ngrid=10)
			# print("SAVED!")
			exit()
		else:
			all_inputs = torch.cat([input_x, inputs_u, inputs_u2], dim=0)

			idx = torch.randperm(all_inputs.size(0))
			input_a, input_b = all_inputs, all_inputs[idx]
			mixed_input = ratio * input_a + (1 - ratio) * input_b

		all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
		target_a, target_b = all_targets, all_targets[idx]

		mixed_target = ratio * target_a + (1 - ratio) * target_b  # need to think how to align it with slerp
		# interleave labeled and unlabed samples between batches to get correct batchnorm calculation
		mixed_input = list(torch.split(mixed_input, batch_size))
		mixed_input = interleave(mixed_input, batch_size)

		logits = [model(mixed_input[0])]
		for input in mixed_input[1:]:
			logits.append(model(input))

		# put interleaved samples back
		logits = interleave(logits, batch_size)
		logits_x = logits[0]
		logits_u = torch.cat(logits[1:], dim=0)

		Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
		                      epoch + batch_idx / args.val_iteration)

		loss = Lx + w * Lu

		# record loss
		losses.update(loss.item(), idx_x.size(0))
		losses_x.update(Lx.item(), idx_x.size(0))
		losses_u.update(Lu.item(), idx_x.size(0))
		ws.update(w, idx_x.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		ema_optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# plot progress
		bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
			batch=batch_idx + 1,
			size=args.val_iteration,
			data=data_time.avg,
			bt=batch_time.avg,
			total=bar.elapsed_td,
			eta=bar.eta_td,
			loss=losses.avg,
			loss_x=losses_x.avg,
			loss_u=losses_u.avg,
			w=ws.avg,
		)
		bar.next()
	bar.finish()

	return (losses.avg, losses_x.avg, losses_u.avg,)


def validate(valloader, model, criterion, epoch, use_cuda, mode):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	bar = Bar(f'{mode}', max=len(valloader))
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(valloader):
			# measure data loading time
			data_time.update(time.time() - end)

			if use_cuda:
				inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
			# compute output
			outputs = model(inputs)
			loss = criterion(outputs, targets)

			# measure accuracy and record loss
			prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
			losses.update(loss.item(), inputs.size(0))
			top1.update(prec1.item(), inputs.size(0))
			top5.update(prec5.item(), inputs.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			# plot progress
			bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
				batch=batch_idx + 1,
				size=len(valloader),
				data=data_time.avg,
				bt=batch_time.avg,
				total=bar.elapsed_td,
				eta=bar.eta_td,
				loss=losses.avg,
				top1=top1.avg,
				top5=top5.avg,
			)
			bar.next()
		bar.finish()
	return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
	filepath = os.path.join(checkpoint, filename)
	torch.save(state, filepath)
	if is_best:
		shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def linear_rampup(current, rampup_length=args.epochs):
	if rampup_length == 0:
		return 1.0
	else:
		current = np.clip(current / rampup_length, 0.0, 1.0)
		return float(current)


class SemiLoss(object):
	def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
		probs_u = torch.softmax(outputs_u, dim=1)

		Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
		Lu = torch.mean((probs_u - targets_u) ** 2)

		return Lx, Lu, args.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
	def __init__(self, model, ema_model, alpha=0.999):
		self.model = model
		self.ema_model = ema_model
		self.alpha = alpha
		self.params = list(model.state_dict().values())
		self.ema_params = list(ema_model.state_dict().values())
		self.wd = 0.02 * args.lr

		for param, ema_param in zip(self.params, self.ema_params):
			param.data.copy_(ema_param.data)

	def step(self):
		one_minus_alpha = 1.0 - self.alpha
		for param, ema_param in zip(self.params, self.ema_params):
			ema_param = ema_param.float()
			param = param.float()
			ema_param.mul_(self.alpha)
			ema_param.add_(param * one_minus_alpha)
			# customized weight decay
			param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
	groups = [batch // (nu + 1)] * (nu + 1)
	for x in range(batch - sum(groups)):
		groups[-x - 1] += 1
	offsets = [0]
	for g in groups:
		offsets.append(offsets[-1] + g)
	assert offsets[-1] == batch
	return offsets


def interleave(xy, batch):
	nu = len(xy) - 1
	offsets = interleave_offsets(batch, nu)
	xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
	for i in range(1, nu + 1):
		xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
	return [torch.cat(v, dim=0) for v in xy]


if __name__ == '__main__':
	main()
