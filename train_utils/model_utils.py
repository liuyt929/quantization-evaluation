import torch,transformers,sys,os,torch.nn as nn,geoopt,typing,utils,transformers,tqdm,math,models
import utils.hadamard_utils as hadamard_utils
from utils.hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
from fast_hadamard_transform import hadamard_transform
from utils.fuse_norm_utils import fuse_ln_linear
from collections import defaultdict
from train_utils.quant_layer import QuantDecoderLayer,QuantRMSNorm,QuantLinear,QuantEmbedding,Quantizer
from transformers import AutoTokenizer
from data_normalization import smooth_utils,rotation_utils
# from transformers import LlamaForCausalLM
from models.modeling_llama import LlamaForCausalLM
from utils.utils import DEV,rdtype,Rdtype
from utils import quant_utils
from geoopt.manifolds import EuclideanStiefel,Stiefel
import torch.nn.functional as F
def skip(*args,**kwargs):
    pass
     
def get_model(
    model_name,args,ptq_args,hf_token=None
):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model:transformers.AutoModelForCausalLM
    if "llama" in model_name.lower():
        config = transformers.LlamaConfig.from_pretrained(model_name,cache_dir='../')
    elif "qwen2" in model_name.lower():
        config= transformers.Qwen2Config.from_pretrained(model_name,cache_dir='../')
    config._attn_implementation = "sdpa" if args.use_sdpa else "eager" 
    dtype = torch.bfloat16 
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    if "qwen2" in model_name.lower():
        model = models.Qwen2ForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map='cpu',
                use_auth_token=hf_token,
                low_cpu_mem_usage=True,
                config = config,
                cache_dir='../'
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,use_fact=True,add_eos_token=False,add_bos_token=False,padding_side="right")
    elif "llama" in model_name.lower():
        model = models.LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map='cpu',
            use_auth_token=hf_token,
            low_cpu_mem_usage=True,
            config = config,
            # cache_dir='/media/DATA-SSD/liuyutong/allQuant'
        )
        tokenizer = transformers.LlamaTokenizerFast.from_pretrained(model_name,use_fact=True,add_eos_token=False,add_bos_token=False,padding_side="right")
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.lm_head.weight.to('cpu')    
    model.seqlen = 2048
    if "llama" in model_name.lower():
        from train_utils.quant_layer import QuantDecoderLayer,QuantRMSNorm,QuantLinear,QuantEmbedding
    elif "qwen2" in model_name.lower():
        from train_utils.quant_layer_qwen import QuantDecoderLayer,QuantRMSNorm,QuantLinear,QuantEmbedding
    
    layers = model.model.layers
    for i in range(len(layers)):
        layers[i] = QuantDecoderLayer(model.config,layers[i],args,ptq_args)
    embed_quant_params = dict(bits=16, sym=not ptq_args.a_asym, dynamic=True, dynamic_method="perchannel")
    model.model.embed_tokens = QuantEmbedding(model.model.embed_tokens,embed_quant_params) 
    model.model.norm = QuantRMSNorm(model.model.norm,dict(bits=32))
    model.lm_head = QuantLinear(model.lm_head,dict(bits=32), name="head") 
    utils.utils.cleanup_memory(False)
    
    return model,tokenizer


class LM:
    # model:LlamaForCausalLM
    def __init__(self,args,ptq_args):
        self.args = args
        self.model_name=args.input_model
        self.model,self.tokenizer = get_model(self.model_name,args,ptq_args)
        
        self.seqlen = 2048
        self.calibrated = False 

    def set_quant_state(self,use_weight_quant:bool=False,use_act_quant:bool=False,use_fully_quant:bool=False):
        model = self.model
        model.model.norm.use_act_quant = use_fully_quant 
        model.model.embed_tokens.use_act_quant = use_fully_quant 
        # model.use_weight_quant=use_weight_quant #additionally
        # use_weight_quant=False #additionally 因为在model的forward函数中统一的weightquant了，所以暂时将layer中的weightquant设置为false
        for layer in model.model.layers: 
            layer.set_quant_state(use_weight_quant,use_act_quant,use_fully_quant)
            
    def set_temporary(self,temporary:bool):
        model = self.model
        model.temporary = temporary
        model.model.temporary = temporary
        model.model.embed_tokens.temporary = temporary
        model.lm_head.temporary = temporary
        for layer in model.model.layers:
            layer.set_temporary(temporary)

    @torch.no_grad()
    def generate_rotate_parameters(self,args=None,ptq_args=None):
        utils.utils.cleanup_memory(False)
        if args is None:
            args = self.args

        model = self.model
        config = model.config
        if hasattr(config,"num_key_value_heads"):
            num_heads = config.num_key_value_heads #4
        else:
            num_heads = config.num_attention_heads
            
        model_dim = config.hidden_size
        head_dim = model_dim // config.num_attention_heads

        
        R_res_value = random_hadamard_matrix(model.config.hidden_size, "cuda") if (not args.use_klt) else klt_R_res(model)
        R_res_module = rotation_utils.RotateModule(R_res_value)
        model.R_res = R_res_module

        layers = model.model.layers
        layers:nn.ModuleList
                
        for idx,layer in enumerate(tqdm.tqdm(layers,unit="layer",desc="Generate Rotating Parameters")):
            layer:nn.Module
            
            R_S_modules = {}
            if args.smooth_up_down:
                S_up_down = torch.ones(layer.mlp.up_proj.weight.shape[0])
                S_up_down_module = smooth_utils.SmoothModule(S_up_down)
                R_S_modules["S_up_down"] = S_up_down_module
            else:
                R_S_modules["S_up_down"] = None
                
            if args.smooth_up_gate:                
                S_up_gate = torch.ones(layer.mlp.gate_proj.weight.shape[0])
                S_up_gate_module = smooth_utils.SmoothModule(S_up_gate)
                R_S_modules["S_up_gate"] = S_up_gate_module
            else:
                R_S_modules["S_up_gate"] = None
                
            if args.smooth_qk:
                S_qk = torch.ones(layer.self_attn.k_proj.weight.shape[0])
                S_qk_module = smooth_utils.SmoothModule(S_qk)
                R_S_modules["S_qk"] = S_qk_module
            else:
                R_S_modules["S_qk"] = None
                
            if args.smooth_ov:
                S_ov = torch.ones(layer.self_attn.v_proj.weight.shape[0])
                S_ov_module = smooth_utils.SmoothModule(S_ov)
                R_S_modules["S_ov"] = S_ov_module
            else:
                R_S_modules["S_ov"] = None

            if args.online_hadamard == 'all': 
                
                mul = layer.mlp.mul 
                had_K,K = hadamard_utils.get_hadK(config.intermediate_size//args.rotate_down_dim)
                mul.down_dim = args.rotate_down_dim
                mul.online_full_had = True
                mul.had_K = had_K
                mul.K = K
                mul.fp32_had = ptq_args.fp32_had
                hadamard_utils.apply_exact_had_to_linear(layer.mlp.down_proj,had_dim=-1,output=False,down_proj_dim = args.rotate_down_dim) 

                
                pv_matmul = layer.self_attn.pv_matmul 
                had_K, K = hadamard_utils.get_hadK(num_heads)
                pv_matmul.online_partial_had = True
                pv_matmul.had_K = had_K
                pv_matmul.K = K
                pv_matmul.head_dim = head_dim
                pv_matmul.fp32_had = ptq_args.fp32_had
                apply_exact_had_to_linear(layer.self_attn.v_proj, had_dim=head_dim, output=True) 
                apply_exact_had_to_linear(layer.self_attn.o_proj, had_dim=-1, output=False) 

            elif args.online_hadamard == 'v': 
                
                pv_matmul = layer.self_attn.pv_matmul 
                had_K, K = hadamard_utils.get_hadK(num_heads)
                pv_matmul.online_partial_had = True
                pv_matmul.had_K = had_K
                pv_matmul.K = K
                pv_matmul.head_dim = head_dim
                pv_matmul.fp32_had = ptq_args.fp32_had
                apply_exact_had_to_linear(layer.self_attn.v_proj, had_dim=head_dim, output=True) 
                apply_exact_had_to_linear(layer.self_attn.o_proj, had_dim=-1, output=False) 

            elif args.online_hadamard == 'down': 
                mul = layer.mlp.mul 
                had_K,K = hadamard_utils.get_hadK(config.intermediate_size//args.rotate_down_dim)
                mul.down_dim = args.rotate_down_dim
                mul.online_full_had = True
                mul.had_K = had_K
                mul.K = K
                mul.fp32_had = ptq_args.fp32_had
                hadamard_utils.apply_exact_had_to_linear(layer.mlp.down_proj,had_dim=-1,output=False) 

            
            if ptq_args.k_bits < 16 and args.online_qk_hadamard: 
                layer.self_attn.ropek.use_hadamard_transform = True
                layer.self_attn.ropeq.use_hadamard_transform = True

            if args.rotate_post_rope:
                post_rope_Q = random_hadamard_matrix(head_dim,"cuda").float()
                layer.self_attn.post_rope_Q = rotation_utils.RotateModule(post_rope_Q)

            if args.rotate_pre_rope:
                pre_rope_Q = random_hadamard_matrix(head_dim,"cuda").float()
                layer.self_attn.pre_rope_Q = rotation_utils.RotateModule(pre_rope_Q)

            
            if args.rotate_ov: 
                if not args.use_klt:
                    R_ov = [random_hadamard_matrix(head_dim, "cuda") for _ in range(num_heads)]
                    R_ov_module = rotation_utils.RotateModule(None,*R_ov)
                else:
                    R_ov = klt_ov(layer,head_dim,num_heads) # 4
                    R_ov_module = rotation_utils.RotateModule(None,*R_ov)
                
                R_S_modules["R_ov"] = R_ov_module
            else:
                R_S_modules["R_ov"] = None

            if args.smooth_norm_linear:
                S_norm_qkv = nn.Parameter(torch.ones(layer.input_layernorm.weight.shape[-1],dtype=torch.float32,device="cuda"))
                S_norm_upgate = nn.Parameter(torch.ones(layer.post_attention_layernorm.weight.shape[0],dtype=torch.float32,device="cuda"))
                S_norm_qkv_module = smooth_utils.SmoothModule(S_norm_qkv)
                S_norm_upgate_module = smooth_utils.SmoothModule(S_norm_upgate)
                R_S_modules["S_norm_qkv"] = S_norm_qkv_module
                R_S_modules["S_norm_upgate"] = S_norm_upgate_module
            
            layer.R_S_modules = nn.ModuleDict(R_S_modules)
            utils.utils.cleanup_memory(False)
        
        self.set_temporary(True)  
            
        utils.utils.cleanup_memory(False)

    
    @torch.no_grad()
    def rotate_smooth_model_inplace(self,training_args,args=None,del_parameters=True):
        utils.utils.cleanup_memory(False)
        if args is None:
            args = self.args
        
        self._rotate_embedding_inplace(self.model)
        
        self._rotate_head_inplace(self.model)
        
        for idx,layer in enumerate(tqdm.tqdm(self.model.model.layers,desc="rotate_inplacing")):
            self._rotate_layer_inplace(self.model,layer,training_args,args)
            if del_parameters: 
                del layer.R_S_modules
                layer.R_S_modules = dict()
        
        
        
        self.set_temporary(False)
        
    
    def _rotate_embedding_inplace(self,model):
        embed_tokens = model.model.embed_tokens
        embed_tokens.temporary = False
        RD = torch.float64

        R_res = model.R_res
        embed_tokens.weight.data = ((embed_tokens.weight.to(device=DEV,dtype=RD)) @ (R_res.weight.to(dtype=RD,device=utils.utils.DEV))).to(dtype=embed_tokens.weight.dtype,device=embed_tokens.weight.device)
        embed_tokens.R_res = None

    def _rotate_head_inplace(self,model):
        head = model.lm_head
        R_res = self.model.R_res.weight
        RD = torch.float64
        head.weight.data = torch.matmul(head.weight.to(dtype=RD,device=DEV),R_res.to(dtype=RD,device=DEV)).to(dtype=head.weight.dtype,device=head.weight.device)
        head.R_res = None
        head.free_temporary()

    
    def _rotate_layer_inplace(self,model,layer:QuantDecoderLayer,training_args,args):
        ori_dev = layer.self_attn.q_proj.weight.device
        RD = torch.float64
        layer.to(DEV)
        
        config = model.config
        if hasattr(config,"num_key_value_heads"):
            num_kv_heads = config.num_key_value_heads
        else:
            num_kv_heads = config.num_attention_heads
        num_attn_heads = config.num_attention_heads    
        model_dim = config.hidden_size
        head_dim = model_dim // config.num_attention_heads
        R_S_modules = layer.R_S_modules
        q,k,v,o = layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj,layer.self_attn.o_proj
        up,gate,down = layer.mlp.up_proj,layer.mlp.gate_proj,layer.mlp.down_proj
        
        x = torch.randn(1,3,4096).to(device=up.weight.device,dtype=up.weight.dtype)
            
        ori_dtype = q.weight.dtype
        
        
        if training_args.smooth_up_down:
            
            S_up_down = R_S_modules["S_up_down"].weight
            up.weight.data = (up.weight.to(RD) / (S_up_down.to(RD).view(-1,1)))
            if up.bias is not None:
                up.bias.data = (up.bias.to(RD) / (S_up_down.to(RD).view(-1)))
            down.weight.data =  (down.weight.to(RD) * (S_up_down.to(RD).view(1,-1)))
        
        
        if training_args.smooth_up_gate:
            
            S_up_gate = R_S_modules["S_up_gate"].weight
            up.weight.data = (up.weight.to(RD) / (S_up_gate.to(RD).view(-1,1)))
            if up.bias is not None:
                up.bias.data = (up.bias.to(RD) / (S_up_gate.to(RD).view(-1)))
            gate.weight.data = (gate.weight.to(RD) * (S_up_gate.to(RD).view(-1,1)))
            if gate.bias is not None:
                gate.bias.data = (gate.bias.to(RD) * (S_up_gate.to(RD).view(-1)))
            layer.mlp.silu.smooth.data = S_up_gate.to(device=up.weight.device,dtype=ori_dtype)
            # with torch.no_grad():
            #     layer.mlp.silu.smooth.copy_(S_up_gate.to(device=up.weight.device, dtype=ori_dtype))

            
        
        if training_args.smooth_qk:
            
            S_qk = R_S_modules["S_qk"].weight
            k.weight.data = (k.weight.to(RD) * (S_qk.to(RD).view(-1,1)))
            if k.bias is not None:
                k.bias.data = (k.bias.to(RD) * (S_qk.to(RD).view(-1)))
                
            if q.weight.shape[0] > S_qk.numel(): 
                S_qk = S_qk.reshape(layer.self_attn.num_key_value_heads,-1)
                n_head,d = S_qk.shape
                S_qk = S_qk[:,None,:].expand(n_head,layer.self_attn.num_key_value_groups,d).reshape(-1)
            q.weight.data = (q.weight.to(RD) / (S_qk.to(RD).view(-1,1)))
            if q.bias is not None:
                q.bias.data = (q.bias.to(RD) / (S_qk.to(RD).view(-1)))
                
        
        if training_args.smooth_ov:
            
            S_ov = R_S_modules["S_ov"].weight
            v.weight.data = (v.weight.to(RD) / (S_ov.to(RD).view(-1,1)))
            if v.bias is not None:
                v.bias.data = (v.bias.to(RD) / (S_ov.to(RD).view(-1)))
                
            if o.weight.shape[-1] > S_ov.numel():
                S_ov = S_ov.reshape(layer.self_attn.num_key_value_heads,-1)
                n_head,d = S_ov.shape
                S_ov = S_ov[:,None,:].expand(n_head,layer.self_attn.num_key_value_groups,d).reshape(-1)
            o.weight.data = (o.weight.to(RD) * (S_ov.to(RD).view(1,-1)))

        
        R_res = model.R_res.weight
        for m in [q,k,v,up,gate]:
            m.weight.data = ((m.weight.to(RD)) @ (R_res.to(dtype=RD,device=m.weight.device)))
            m.free_temporary()
        for m in [down,o]:
            m.weight.data = torch.matmul(R_res.T.to(dtype=RD,device=m.weight.device),m.weight.to(RD))
            if m.bias is not None:
                m.bias.data = torch.matmul(R_res.T.to(dtype=RD,device=m.bias.device),m.bias.to(RD))
            m.free_temporary()

        
        if training_args.rotate_pre_rope: 
            pre_rope_Q = layer.self_attn.pre_rope_Q.weight
            layer.self_attn.ropek.temporary = True
            layer.self_attn.ropek.temporary = True
            for m in [q,k]:
                ic = m.weight[-1]
                m.weight.data = torch.matmul(m.weight.T.reshape(ic,-1,head_dim).to(RD),pre_rope_Q.to(dtype=RD,device=m.weight.device) \
                                                .unsqueeze(0)).reshape(m.weight.shape).to(m.weight.dtype).T 
                if m.bias is not None:
                    m.bias = torch.matmul(m.bias.to(dtype=RD).reshape(-1,head_dim),pre_rope_Q.to(dtype=RD,device=m.bias.device)) \
                                                .reshape(m.bias.shape).to(m.bias.dtype) 

                
                
                m.free_temporary()
        if training_args.rotate_post_rope:
            layer.self_attn.ropeq.temporary = True
            layer.self_attn.ropek.temporary = True
            

        
        if training_args.rotate_ov:
            
            R_ov = torch.stack(list(R_S_modules["R_ov"].weight),dim=0)
            v,o = layer.self_attn.v_proj,layer.self_attn.o_proj
            
            v.weight.data = (v.weight.T.reshape(-1,num_kv_heads,head_dim).unsqueeze(2).to(RD) @ R_ov.to(RD)).reshape(-1,num_kv_heads*head_dim).T
            if v.bias is not None:
                
                v.bias.data = (v.bias.reshape(num_kv_heads,head_dim).unsqueeze(2).to(RD) @ R_ov.to(RD)).reshape(v.bias.shape)
            
            
            if layer.self_attn.num_key_value_groups != 1:
                h, d1, d2 = R_ov.shape
                repeated_R_ov = R_ov[:,None,:,:].expand(h, layer.self_attn.num_key_value_groups, d1, d2).reshape(-1, d1, d2)
                o.weight.data = (o.weight.reshape(-1,num_attn_heads,head_dim).unsqueeze(2).to(RD) @ repeated_R_ov.to(RD)).reshape(o.weight.shape)
            else:
                o.weight.data = (o.weight.reshape(-1,num_attn_heads,head_dim).unsqueeze(2).to(RD) @ R_ov.to(RD)).reshape(o.weight.shape)
        
        
        if training_args.smooth_norm_linear:
            S_norm_qkv = R_S_modules["S_norm_qkv"].weight
            layer.input_layernorm.weight.data = (layer.input_layernorm.weight.to(RD) / S_norm_qkv.to(RD))
            for m in [q,k,v]:
                m.weight.data = (m.weight.to(RD) * S_norm_qkv.to(RD))
            S_norm_upgate = R_S_modules['S_norm_upgate'].weight
            layer.post_attention_layernorm.weight.data = (layer.post_attention_layernorm.weight.to(RD) / S_norm_upgate.to(RD))
            for m in [up,gate]:
                m.weight.data = (m.weight.to(RD) * S_norm_upgate.to(RD))
            
        for m in [q,k,v,o,up,gate,down,layer.input_layernorm,layer.post_attention_layernorm]:
            m.weight.data = m.weight.data.to(ori_dtype)
            if hasattr(m,"bias") and m.bias is not None:
                m.bias.data = m.bias.data.to(ori_dtype)
        
        # layer.to(ori_dev)
        # layer.to('cpu')
    
    def set_dynamic(self,dynamic=True):
        for m in self.model.modules():
            if isinstance(m,Quantizer):
                m.enable_dynamic(dynamic)
    
    def set_observer(self,observe=True):
        for m in self.model.modules():
            if isinstance(m,Quantizer):
                m.enable_observer(observe)

    def only_q1(self,):
        layers = self.model.model.layers
        for layer in layers:
            layer.input_layernorm.use_act_quant = True
            layer.post_attention_layernorm.use_act_quant = True


def klt_R_res(model):
    
    layers = model.model.layers
    params = list()
    for layer in layers:
        q,k,v,up,gate = layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj,layer.mlp.up_proj,layer.mlp.gate_proj
        for m in (q,k,v,up,gate):
            params.append(m.weight)
    params = torch.cat(params,dim=0)
    params = params-params.mean(dim=0)
    
    cov_matrix = torch.cov(params.float().T) 
    eigs,eiv =  torch.linalg.eigh(cov_matrix)
    H = hadamard_utils.random_hadamard_matrix(params.size(-1),params.device) 
    return (eiv.double()@H).float()

def klt_ov(layer,head_dim=128,kv_heads=8):  
    o,v = layer.self_attn.o_proj,layer.self_attn.v_proj 
    oc,ic = o.weight.shape
    num_kvg = layer.self_attn.num_key_value_groups
    flat_wo  = o.weight.reshape(oc,-1,num_kvg,head_dim)
    ret = list()
    for i in range(kv_heads):
        params = flat_wo[:,i].reshape(-1,head_dim) 
        params = params-params.mean(dim=0) 
        cov_matrix = torch.cov(params.float().T)
        eigs,eiv = torch.linalg.eigh(cov_matrix)
        H = hadamard_utils.random_hadamard_matrix(params.size(-1),o.weight.device)
        ret.append((eiv.double()@H).float())
    return ret

def rotate_smooth_model_inplace(model,st_path,training_args,args=None,del_parameters=True):
    utils.utils.cleanup_memory(False)
    st = torch.load(st_path, map_location="cuda")
    R_res=st["R_res.weight"]
    rotate_embedding_inplace(model,R_res)
    config = model.config
    if hasattr(config,"num_key_value_heads"):
        num_kv_heads = config.num_key_value_heads
    else:
        num_kv_heads = config.num_attention_heads
    rotate_head_inplace(model,R_res)
    for idx,layer in enumerate(tqdm.tqdm(model.model.layers,desc="rotate_inplacing")):
        R_S_modules={}
        hadamard_utils.apply_exact_had_to_linear(layer.mlp.down_proj,had_dim=-1,output=False)
        if training_args.smooth_ov:
            R_S_modules['S_ov']=st['model.layers.'+str(idx)+'.R_S_modules.S_ov.weight']
        if training_args.smooth_up_down:
            R_S_modules['S_up_down']=st['model.layers.'+str(idx)+'.R_S_modules.S_up_down.weight']
        if training_args.smooth_norm_linear:
            R_S_modules['S_norm_qkv']=st['model.layers.'+str(idx)+'.R_S_modules.S_norm_qkv.weight']
            R_S_modules['S_norm_upgate']=st['model.layers.'+str(idx)+'.R_S_modules.S_norm_upgate.weight']
        if training_args.rotate_ov:
            R_list=[]
            for i in range(num_kv_heads):
                R_list.append(st['model.layers.'+str(idx)+'.R_S_modules.R_ov.weight.'+str(i)])
            R_S_modules['R_ov']=R_list
        
        rotate_layer_inplace(model,R_S_modules,R_res,layer,training_args)
        del R_S_modules
        torch.cuda.empty_cache()


    

def rotate_embedding_inplace(model,R_res):
    embed_tokens = model.model.embed_tokens
    embed_tokens.temporary = False
    RD = torch.float64

    embed_tokens.weight.data = ((embed_tokens.weight.to(device=DEV,dtype=RD)) @ (R_res.to(dtype=RD,device=utils.utils.DEV))).to(dtype=embed_tokens.weight.dtype,device=embed_tokens.weight.device)

def rotate_head_inplace(model,R_res):
    head = model.lm_head
    RD = torch.float64
    head.weight.data = torch.matmul(head.weight.to(dtype=RD,device=DEV),R_res.to(dtype=RD,device=DEV)).to(dtype=head.weight.dtype,device=head.weight.device)


def rotate_layer_inplace(model,R_S_modules,R_res,layer:QuantDecoderLayer,training_args):
    ori_dev = layer.self_attn.q_proj.weight.device
    RD = torch.float64
    layer.to(DEV)
    
    config = model.config
    if hasattr(config,"num_key_value_heads"):
        num_kv_heads = config.num_key_value_heads
    else:
        num_kv_heads = config.num_attention_heads
    num_attn_heads = config.num_attention_heads    
    model_dim = config.hidden_size
    head_dim = model_dim // config.num_attention_heads
    q,k,v,o = layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj,layer.self_attn.o_proj
    up,gate,down = layer.mlp.up_proj,layer.mlp.gate_proj,layer.mlp.down_proj
        
    ori_dtype = q.weight.dtype
    
    
    if training_args.smooth_up_down:
        
        S_up_down = R_S_modules["S_up_down"]
        up.weight.data = (up.weight.to(RD) / (S_up_down.to(RD).view(-1,1)))
        if up.bias is not None:
            up.bias.data = (up.bias.to(RD) / (S_up_down.to(RD).view(-1)))
        down.weight.data =  (down.weight.to(RD) * (S_up_down.to(RD).view(1,-1)))
    
    
    if training_args.smooth_up_gate:
        
        S_up_gate = R_S_modules["S_up_gate"]
        up.weight.data = (up.weight.to(RD) / (S_up_gate.to(RD).view(-1,1)))
        if up.bias is not None:
            up.bias.data = (up.bias.to(RD) / (S_up_gate.to(RD).view(-1)))
        gate.weight.data = (gate.weight.to(RD) * (S_up_gate.to(RD).view(-1,1)))
        if gate.bias is not None:
            gate.bias.data = (gate.bias.to(RD) * (S_up_gate.to(RD).view(-1)))
        layer.mlp.silu.smooth.data = S_up_gate.to(device=up.weight.device,dtype=ori_dtype)
        # with torch.no_grad():
        #     layer.mlp.silu.smooth.copy_(S_up_gate.to(device=up.weight.device, dtype=ori_dtype))

        
    
    if training_args.smooth_qk:
        
        S_qk = R_S_modules["S_qk"]
        k.weight.data = (k.weight.to(RD) * (S_qk.to(RD).view(-1,1)))
        if k.bias is not None:
            k.bias.data = (k.bias.to(RD) * (S_qk.to(RD).view(-1)))
            
        if q.weight.shape[0] > S_qk.numel(): 
            S_qk = S_qk.reshape(layer.self_attn.num_key_value_heads,-1)
            n_head,d = S_qk.shape
            S_qk = S_qk[:,None,:].expand(n_head,layer.self_attn.num_key_value_groups,d).reshape(-1)
        q.weight.data = (q.weight.to(RD) / (S_qk.to(RD).view(-1,1)))
        if q.bias is not None:
            q.bias.data = (q.bias.to(RD) / (S_qk.to(RD).view(-1)))
            
    
    if training_args.smooth_ov:
        
        S_ov = R_S_modules["S_ov"]
        v.weight.data = (v.weight.to(RD) / (S_ov.to(RD).view(-1,1)))
        if v.bias is not None:
            v.bias.data = (v.bias.to(RD) / (S_ov.to(RD).view(-1)))
            
        if o.weight.shape[-1] > S_ov.numel():
            S_ov = S_ov.reshape(layer.self_attn.num_key_value_heads,-1)
            n_head,d = S_ov.shape
            S_ov = S_ov[:,None,:].expand(n_head,layer.self_attn.num_key_value_groups,d).reshape(-1)
        o.weight.data = (o.weight.to(RD) * (S_ov.to(RD).view(1,-1)))

    for m in [q,k,v,up,gate]:
        m.weight.data = ((m.weight.to(RD)) @ (R_res.to(dtype=RD,device=m.weight.device)))
    for m in [down,o]:
        m.weight.data = torch.matmul(R_res.T.to(dtype=RD,device=m.weight.device),m.weight.to(RD))
        if m.bias is not None:
            m.bias.data = torch.matmul(R_res.T.to(dtype=RD,device=m.bias.device),m.bias.to(RD))

    
    if training_args.rotate_pre_rope: 
        pre_rope_Q = layer.self_attn.pre_rope_Q
        layer.self_attn.ropek.temporary = True
        layer.self_attn.ropek.temporary = True
        for m in [q,k]:
            ic = m.weight[-1]
            m.weight.data = torch.matmul(m.weight.T.reshape(ic,-1,head_dim).to(RD),pre_rope_Q.to(dtype=RD,device=m.weight.device) \
                                            .unsqueeze(0)).reshape(m.weight.shape).to(m.weight.dtype).T 
            if m.bias is not None:
                m.bias = torch.matmul(m.bias.to(dtype=RD).reshape(-1,head_dim),pre_rope_Q.to(dtype=RD,device=m.bias.device)) \
                                            .reshape(m.bias.shape).to(m.bias.dtype) 


    
    if training_args.rotate_ov:
        
        R_ov = torch.stack(R_S_modules["R_ov"],dim=0)
        v,o = layer.self_attn.v_proj,layer.self_attn.o_proj
        
        v.weight.data = (v.weight.T.reshape(-1,num_kv_heads,head_dim).unsqueeze(2).to(RD) @ R_ov.to(RD)).reshape(-1,num_kv_heads*head_dim).T
        if v.bias is not None:
            
            b=v.bias     
        # v_proj.bias.data = (b.reshape(num_kv_heads,head_dim).unsqueeze(2) @ R_ov).reshape(v_proj.bias.shape)
            b = b.reshape(num_kv_heads, head_dim).to(RD)  # (2, 128)
            v.bias.data = (b.unsqueeze(1) @ R_ov.to(RD)).squeeze(1).reshape(-1)
            # v.bias.data = (v.bias.reshape(num_kv_heads,head_dim).unsqueeze(2).to(RD) @ R_ov.to(RD)).reshape(v.bias.shape)
        
        
        if layer.self_attn.num_key_value_groups != 1:
            h, d1, d2 = R_ov.shape
            repeated_R_ov = R_ov[:,None,:,:].expand(h, layer.self_attn.num_key_value_groups, d1, d2).reshape(-1, d1, d2)
            o.weight.data = (o.weight.reshape(-1,num_attn_heads,head_dim).unsqueeze(2).to(RD) @ repeated_R_ov.to(RD)).reshape(o.weight.shape)
        else:
            o.weight.data = (o.weight.reshape(-1,num_attn_heads,head_dim).unsqueeze(2).to(RD) @ R_ov.to(RD)).reshape(o.weight.shape)
    
    
    if training_args.smooth_norm_linear:
        S_norm_qkv = R_S_modules["S_norm_qkv"]
        layer.input_layernorm.weight.data = (layer.input_layernorm.weight.to(RD) / S_norm_qkv.to(RD))
        for m in [q,k,v]:
            m.weight.data = (m.weight.to(RD) * S_norm_qkv.to(RD))
        S_norm_upgate = R_S_modules['S_norm_upgate']
        layer.post_attention_layernorm.weight.data = (layer.post_attention_layernorm.weight.to(RD) / S_norm_upgate.to(RD))
        for m in [up,gate]:
            m.weight.data = (m.weight.to(RD) * S_norm_upgate.to(RD))
        
    for m in [q,k,v,o,up,gate,down,layer.input_layernorm,layer.post_attention_layernorm]:
        m.weight.data = m.weight.data.to(ori_dtype)
        if hasattr(m,"bias") and m.bias is not None:
            m.bias.data = m.bias.data.to(ori_dtype)
    layer.to(ori_dev)




