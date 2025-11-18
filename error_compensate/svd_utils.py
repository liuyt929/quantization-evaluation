import torch
from utils.quant_utils import ActQuantWrapper, WeightQuantizer

class LayerSvdCompute:
    """ 直接对 ActQuantWrapper 操作，计算低秩分解 """

    @staticmethod
    def rtn_compute_qw(W, layer_weight_bits, w_groupsize, w_clip, w_asym):
        """ 计算量化后的权重 """
        quantizer = WeightQuantizer()
        quantizer.configure(
            layer_weight_bits,
            perchannel=True,
            sym=not w_asym,
            mse=w_clip,
            weight_groupsize=w_groupsize,
        )
        quantizer.find_params(W)
        q, int_weight, scale = quantizer.fake_quantize(W)
        # print(q)
        return q

    @staticmethod
    @torch.no_grad()
    def q_error_T(W, layer_weight_bits, w_groupsize, w_clip, w_asym):
        """ 计算量化误差的转置矩阵 """
        return (W - LayerSvdCompute.rtn_compute_qw(W, layer_weight_bits, w_groupsize, w_clip, w_asym)).transpose(0, 1)
    
    @staticmethod
    @torch.no_grad()
    def approximate(wrapper: ActQuantWrapper, name: str, ptq_args) -> None:
        """ 对权重进行 SVD 近似，并更新 A 和 B """
        W = wrapper.module.weight
        rank = wrapper.rank

        # 处理量化参数
        layer_weight_bits = ptq_args.w_bits
        if ptq_args.int8_down_proj and "down_proj" in name:
            layer_weight_bits = 8
        if ptq_args.export_to_et:
            layer_weight_bits = 8  # Executorch 导出
            w_groupsize = -1
        else:
            w_groupsize = ptq_args.w_groupsize

        w_clip = ptq_args.w_clip
        w_asym = ptq_args.w_asym

        # 计算量化误差
        q_error_T = LayerSvdCompute.q_error_T(W, layer_weight_bits, w_groupsize, w_clip, w_asym)
        # print("q_error_T",q_error_T)
        # 进行 SVD 分解
        U, S, V_T = torch.linalg.svd(q_error_T.float())

        # 低秩截断
        U_k = U[:, :rank]
        S_k = S[:rank]
        V_T_k = V_T[:rank, :]
        
        # print("matmul_result",res)
        # print(q_error_T-res)
        # raise ValueError
        # 更新 wrapper 的 A 和 B
        wrapper.A = torch.nn.Parameter(U_k.to(dtype=torch.bfloat16),requires_grad =False)
        wrapper.B = torch.nn.Parameter((torch.diag(S_k) @ V_T_k).to(dtype=torch.bfloat16),requires_grad =False)

    def approximate_with_qweight(wrapper: ActQuantWrapper, name: str,origin_weight,qweight,scale_rms=None) -> None:
        """ 对权重进行 SVD 近似，并更新 A 和 B """
        rank = wrapper.rank
        # 计算量化误差
        # q_error_T = W^T - W_q^T
        # SVD(S @ q_error_T) = U @ S @ V^T
        # A = S^-1 @ U_k
        if scale_rms is not None:
            scale_rms=scale_rms.to(device=qweight.device,dtype=torch.bfloat16)
            q_error_T=torch.diag(scale_rms) @(origin_weight - qweight).transpose(0, 1)
        else:
            q_error_T=(origin_weight - qweight).transpose(0, 1)
        # print("q_error_T",q_error_T)
        # 进行 SVD 分解
        U, S, V_T = torch.linalg.svd(q_error_T.float())

        # 低秩截断
        U_k = U[:, :rank]
        S_k = S[:rank]
        V_T_k = V_T[:rank, :]
        # print("matmul_result",res)
        # print(q_error_T-res)
        # raise ValueError
        # 更新 wrapper 的 A 和 B
        if scale_rms is not None:
            inv_scale = 1.0 / torch.clamp(scale_rms, min=torch.finfo(scale_rms.dtype).eps)
            scaled_U_k = U_k / inv_scale.unsqueeze(1)  # shape: [C, K]
            wrapper.A = torch.nn.Parameter(scaled_U_k.to(dtype=torch.bfloat16),requires_grad =False)
        else:
            wrapper.A = torch.nn.Parameter(U_k.to(dtype=torch.bfloat16),requires_grad =False)
        wrapper.B = torch.nn.Parameter((torch.diag(S_k) @ V_T_k).to(dtype=torch.bfloat16),requires_grad =False)



class ScaleHookFactoryDiagonal:
    """
    scale = diag( sqrt( E[ x_1^2]), sqrt( E[ x_2^2]), ..., sqrt( E[ x_n^2] ) )
    """

    def __init__(self, torch_dtype) -> None:
        self.scales = {}
        self.n_samples = {}
        self.compute_devices = {}
        self.torch_dtype = torch_dtype
        self.handles = []

    def get_scale_hook(self, name: str) -> callable:
        self.scales[name] = None
        self.n_samples[name] = 0

        @torch.no_grad()
        def scale_hook(
            module: torch.nn.Linear,
            input: tuple[torch.Tensor],
            output: torch.Tensor,
        ) -> None:
            x = input[0]
            x = x.view(-1, x.shape[-1])
            num_samples, _ = x.shape
            x = x.pow(2).sum(0)

            self.n_samples[name] += num_samples
            if self.scales[name] is None:
                self.compute_devices[name] = x.device
                if self.torch_dtype is None:
                    self.torch_dtype = x.dtype
                if self.compute_devices[name].type == "cpu":
                    logger.warning("Using CPU for computing scale, this may be slow")
                scale = x.to(self.torch_dtype)
            else:
                scale = self.scales[name].to(self.compute_devices[name])
                scale = scale + x.to(self.torch_dtype)

            self.scales[name] = scale

        return scale_hook

    @torch.no_grad()
    def get_scale_dict(self, progress_bar=False) -> dict[str, torch.Tensor]:
        scale_names_prog_bar = tqdm(
            self.scales, desc="Computing scale", disable=not progress_bar, total=len(self.scales)
        )

        for name in scale_names_prog_bar:
            # for name in self.scales:
            scale = self.scales[name].to(self.compute_devices[name])
            scale = torch.sqrt(scale) * (1 / math.sqrt(self.n_samples[name]))
            # scale = torch.clamp(scale, min=CLAMP_MIN)
            self.scales[name] = scale

        return self.scales

    def remove_all_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
