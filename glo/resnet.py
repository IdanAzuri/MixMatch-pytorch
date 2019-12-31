import numpy as np
from torch import nn


class Generator(nn.Module):
	def __init__(self, z_dim, size, nfilter=64, nfilter_max=512, **kwargs):
		super(Generator, self).__init__()
		s0 = self.s0 = 4
		nf = self.nf = nfilter
		nf_max = self.nf_max = nfilter_max

		self.z_dim = z_dim

		# Submodules
		nlayers = int(np.log2(size / s0))
		self.nf0 = min(nf_max, nf * 2 ** nlayers)

		self.fc = nn.Linear(z_dim, self.nf0 * s0 * s0)

		blocks = []
		for i in range(nlayers):
			nf0 = min(nf * 2 ** (nlayers - i), nf_max)
			nf1 = min(nf * 2 ** (nlayers - i - 1), nf_max)
			blocks += [
					ResnetBlock(nf0, nf1),
					nn.functional.interpolate(scale_factor=2)
					]

		blocks += [
				ResnetBlock(nf, nf),
				]

		self.resnet = nn.Sequential(*blocks)
		self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

	def forward(self, z):
		batch_size = z.size(0)
		zn = z.norm(2, 1).detach().unsqueeze(1).expand_as(z)
		z = z.div(zn)

		out = self.fc(z)
		out = out.view(batch_size, self.nf0, self.s0, self.s0)

		out = self.resnet(out)

		out = self.conv_img(actvn(out))
		out = nn.functional.tanh(out)

		return out


class ResnetBlock(nn.Module):
	def __init__(self, fin, fout, fhidden=None, is_bias=True):
		super(ResnetBlock, self).__init__()
		# Attributes
		self.is_bias = is_bias
		self.learned_shortcut = (fin != fout)
		self.fin = fin
		self.fout = fout
		if fhidden is None:
			self.fhidden = min(fin, fout)
		else:
			self.fhidden = fhidden

		# Submodules
		self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
		self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
		if self.learned_shortcut:
			self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

	def forward(self, x):
		x_s = self._shortcut(x)
		dx = self.conv_0(actvn(x))
		dx = self.conv_1(actvn(dx))
		out = x_s + 0.1 * dx

		return out

	def _shortcut(self, x):
		if self.learned_shortcut:
			x_s = self.conv_s(x)
		else:
			x_s = x
		return x_s


def actvn(x):
	out = nn.functional.leaky_relu(x, 2e-1)
	return out
