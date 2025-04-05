#!/usr/bin/env python

import torch

import re

class _FunctionCorrelation(torch.autograd.Function):
	@staticmethod
	def forward(self, first, second):
		rbot0 = first.new_zeros([ first.shape[0], first.shape[2] + 8, first.shape[3] + 8, first.shape[1] ])
		rbot1 = first.new_zeros([ first.shape[0], first.shape[2] + 8, first.shape[3] + 8, first.shape[1] ])

		self.save_for_backward(first, second, rbot0, rbot1)

		assert(first.is_contiguous() == True)
		assert(second.is_contiguous() == True)

		output = first.new_zeros([ first.shape[0], 81, first.shape[2], first.shape[3] ])

		# first, second: [N, C, H, W] 형태의 입력 (CPU 상에 위치)
		patches = torch.nn.functional.unfold(second, kernel_size=(9,9), padding=4)  # [N, C*81, H*W]
		patches = patches.view(first.shape[0], first.shape[1], 81, first.shape[2] * first.shape[3])      # 두번째 입력의 모든 9x9 패치 (81 오프셋) 
		first_exp = first.view(first.shape[0], first.shape[1], 1, first.shape[2] * first.shape[3])       # 첫번째 입력을 [N, C, 1, H*W]로 변형
		# 첫번째와 두번째 패치의 elementwise 곱을 channel 차원 합산 -> 상관도 계산
		corr = (first_exp * patches).sum(dim=1)    # [N, 81, H*W]
		output = corr.view(first.shape[0], 81, first.shape[2], first.shape[3])            # [N, 81, H, W] 코스트 볼륨 결과

		return output
	# end
# end

def FunctionCorrelation(tenFirst, tenSecond):
	return _FunctionCorrelation.apply(tenFirst, tenSecond)
# end

class ModuleCorrelation(torch.nn.Module):
	def __init__(self):
		super(ModuleCorrelation, self).__init__()
	# end

	def forward(self, tenFirst, tenSecond):
		return _FunctionCorrelation.apply(tenFirst, tenSecond)
	# end
# end