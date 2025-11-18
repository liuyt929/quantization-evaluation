import torch,torch.nn as nn,torch.nn.functional as F,math
from typing import Union,Literal

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,s,zp,qmin,qmax):
        x_int = ((x/s).round() + zp).clip(qmin,qmax)
        return (x_int - zp) * s
    
    def backward(ctx,grad_output):
        return grad_output,None,None,None,None

def round_ste(x:torch.Tensor):
    return (x.round() - x).detach() + x

class Quantizer(nn.Module):
    def __init__(self,bits:int=16,sym=True,groupsize=-1,clip_ratio=1.0,
                 dynamic=True,dynamic_method:Union[Literal["pertoken"],Literal["pertensor"],Literal["perchannel"]]="pertoken",
                 static_calib_method="minmax",mse=False,lwc=False,shape=None):
        super(Quantizer,self).__init__()
        self.bits = bits
        self.sym = sym
        self.group_size = groupsize
        self.mse = mse
        self.clip_ratio = clip_ratio 
        self.qmax,self.qmin = (2**(bits-1)) -1 if sym else 2**(bits)-1 , -(2**(bits-1)) if sym else 0
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method 
        if self.bits == 16:
            self.dynamic_method == "pertensor"
        
        self.static_calib_method = static_calib_method
        self.is_observing = False
        # self.scale = None
        self.register_buffer('scale', torch.zeros(1)) 
        quant_values = [-6, -5, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
        self.register_buffer('values', torch.tensor(quant_values))
        
        # 计算正确的阈值 (15个阈值对应16个区间)
        thresholds = [(a + b)/2 for a, b in zip(quant_values[:-1], quant_values[1:])]
        self.register_buffer('thresholds', torch.tensor(thresholds))
        self.blocksize=32

        
        self.lwc = lwc
        if self.lwc:
            dim = shape[0]
            if groupsize != -1:
                dim = dim * (shape[-1] % groupsize)
            self.upbound_factor = nn.Parameter(torch.ones(dim,1)*4)
            self.lowbound_factor = nn.Parameter(torch.ones(dim,1,)*4)
    
    def fake_quant(x):
        pass
    
    def forward(self,x):
        if self.bits > 16:
            return x
        x_dtype = x.dtype
        F32_EXP_BIAS = 127
        F32_MIN_NORMAL = 2 ** (-F32_EXP_BIAS + 1)
        epsilon = F32_MIN_NORMAL
        original_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-1] // self.blocksize, self.blocksize)
        absmax_per_block = torch.max(reshaped_x.abs(), dim=-1, keepdim=True)[0]
        absmax_per_block[absmax_per_block == 0] += epsilon

        Elow = 0        # `Elow` is FLEXIBLE, determining `BIAS`
        Ehigh = 2      # `Ehigh` depends on `Elow`
        Mhigh = 1
        Qmax = 6.0
        Qmin = -6.0
        
        # scale_per_block = (2 * absmax_per_block) / (Qmax - Qmin)
        # scale_per_block = scale_per_block.to(x)
        #用最大最小算出scale，然后取log，再clamp8位，这样相当于scale用e8m0存储
        # Microscaling's original setting
        scale_per_block = torch.floor(torch.log2(absmax_per_block)) - Ehigh 
        scale_per_block.clamp_(-127, 127)
        scale_per_block = 2 ** scale_per_block
        
        scale_per_block = scale_per_block.repeat(1, 1, 1, self.blocksize).reshape(original_shape)
        self.scale=scale_per_block
        
        x_scaled = x / self.scale*(3/4)
        # print(x_scaled)
        # 初始化量化输出
        quantized = torch.zeros_like(x_scaled)
        
        # 下边界处理 (x < 最小阈值)
        min_threshold = self.thresholds[0]
        min_value = self.values[0]
        quantized = torch.where(x_scaled < min_threshold, min_value, quantized)
        
        # 中间区间处理
        for i in range(len(self.thresholds)-1):
            left_thresh = self.thresholds[i]
            right_thresh = self.thresholds[i+1]
            q_value = self.values[i+1]
            
            # 区间判断
            in_interval = (x_scaled >= left_thresh) & (x_scaled < right_thresh)
            quantized = torch.where(in_interval, q_value, quantized)
        
        # 上边界处理 (x >= 最大阈值)
        max_threshold = self.thresholds[-1]
        max_value = self.values[-1]
        quantized = torch.where(x_scaled >= max_threshold, max_value, quantized)
        
        # 应用STE (Straight-Through Estimator)
        # 前向：quantized，反向：x_scaled的梯度
        out = quantized + (x_scaled - x_scaled.detach())
        
        # dequant反缩放
        return (out * self.scale* (4/3)).to(x_dtype)
    
        if self.dynamic:
            if self.bits >= 16:
                return x
            s,zp = self.dynamic_quant_params(x)
            
            x_fake = ((round_ste(x/s) + zp).clip(self.qmin,self.qmax) - zp)*s
            return x_fake.to(x_dtype)
        
        else:
            s,zp = self.static_quant_params(x)
            if self.is_observing:
                return x
            return STE.apply(x,s,zp,self.qmin,self.qmax).to(x_dtype)
            x_fake = ((round_ste(x/s) + zp).clip(self.qmin,self.qmax) - zp)*s
            return x_fake.to(x_dtype)
            
    
    
    def dynamic_quant_params(self,x):
        ori_shape = list(x.shape)
        if self.dynamic_method == "pertensor":
            xmax,xmin = torch.max(x),torch.min(x)
        elif self.dynamic_method == "pertoken" or self.group_size != -1:
            if self.group_size != -1:
                reshaped_x = x.reshape(*ori_shape[:-1],ori_shape[-1] // self.group_size, self.group_size)
                xmax,xmin = torch.amax(reshaped_x,dim=-1,keepdim=True),torch.amin(reshaped_x,dim=-1,keepdim=True)
            else:
                xmax,xmin = torch.amax(x,dim=tuple(range(min(2,x.dim()-1),x.dim())),keepdim=True),torch.amin(x,dim=tuple(range(min(2,x.dim()-1),x.dim())),keepdim=True) 
        elif self.dynamic_method == "perchannel": 
            xmax,xmin = torch.amax(x,dim=list(range(0,x.dim()-1)),keepdim=True),torch.amin(x,dim=(0,1),keepdim=False)
        else:
            raise NotImplemented
        if self.lwc:
            xmax,xmin = F.sigmoid(self.upbound_factor)*xmax , F.sigmoid(self.lowbound_factor) * xmin
        xmax = xmax.clip(min=1e-6)
        xmin = xmin.clip(max=0)
        xmax,xmin = xmax*self.clip_ratio, xmin * self.clip_ratio
        if self.mse:
            x = x.detach()
            xmax = xmax.detach()
            xmin = xmin.detach()
            scale= torch.ones_like(xmax)
            zp = torch.zeros_like(xmax)
            best_err = torch.ones_like(xmax).fill_(10000)
            for i in range(80):
                p = 1 - (i/100)
                xmin1 = p * xmin.detach()
                xmax1 = p * xmax.detach()
                if self.sym:
                    s1 = torch.maximum(xmax1/self.qmax ,xmin1/self.qmin)
                    zp1 = torch.zeros_like(s1)
                else:
                    s1 = (xmax1 - xmin1) / self.qmax
                    zp1 = torch.round(-xmin1 / s1)
                q = (((x/s1).round() + zp1).clip(self.qmin,self.qmax) - zp1) * s1 
                err = q.sub_(x.data).pow_(2).sum(-1,keepdim=True)
                tmp = err < best_err
                if torch.any(tmp):
                    scale[tmp] = s1[tmp].to(scale.dtype)
                    zp[tmp] = zp1[tmp].to(zp.dtype)
                    best_err[tmp] = err[tmp].to(best_err.dtype)
        else:
            if self.sym:
                scale = torch.maximum((xmin)/self.qmin ,xmax/(self.qmax))
                
                zp = torch.zeros_like(scale)
            else:
                scale = (xmax - xmin).clip_(1e-6) / self.qmax
                zp = torch.round(-xmin / scale)

        if self.group_size != -1:
            scale = scale.repeat(1, 1, 1, self.group_size).reshape(ori_shape)
            zp = zp.repeat(1, 1, 1, self.group_size).reshape(ori_shape)
        return scale, zp 
    
    def percent(self,x,percent:float):
        if self.dynamic_method == "pertensor":
            return torch.quantile(x.abs(),percent)
        elif self.dynamic_method == "perchannel":
            return torch.quantile(x.abs().reshape(-1,x.shape[-1]),percent,dim=0)
        else:
            raise NotImplemented
    
    def static_quant_params(self,x):
        if not self.is_observing:
            assert self.scale is not None,"must be set before static quantization"
            return self.scale,self.zp
        assert self.sym == True,"only support"
        if self.static_calib_method == "minmax":
            if self.scale is None:
                self.scale,self.zp  = self.dynamic_quant_params(x)
            else:
                scale,self.zp = self.dynamic_quant_params(x)
                self.scale = torch.max(self.scale,scale)
        elif "percent" in self.static_calib_method:
            percent = float(self.static_calib_method[7:])
            if self.scale is None:
                self.scale = self.percnet(x,percent)
                self.zp = torch.zeros_like(self.scale)
            else:
                scale = self.percent(x,percent)
                self.scale = scale * 0.01 + self.scale * 0.99
        return self.scale,self.zp
            
    def enable_dynamic(self,value):
        self.dynamic = value

    def enable_observer(self,value=True):
        self.dynamic = False
        self.is_observing = value
