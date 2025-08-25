import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib.pyplot as plt
import numpy as np

__version__ = "0.0"

import os

from pynvml import *
import subprocess


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

import tensorboard as tb

import wandb

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def tensors2vectors(tensors):
    def tensor2vector(tensor):
        flo = (tensor.permute(1, 2, 0).numpy()-0.5)*1000
        r = 8
        plt.quiver(flo[::-r, ::r, 0], -flo[::-r, ::r, 1], color='r', scale=r*20)
        plt.savefig('temp.jpg')
        plt.clf()
        return plt.imread('temp.jpg').transpose(2, 0, 1)
    return torch.from_numpy(np.array([tensor2vector(tensor) for tensor in tensors])) / 255

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# ==============================================================================
# (1) Define or Import the TypeEmbedder Class
# ==============================================================================
class TypeEmbedder(nn.Module):
    """
    Adds a learnable type embedding to the input tensor.
    """
    def __init__(self, num_types: int, embedding_dim: int):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, embedding_dim)

    def forward(self, x: torch.Tensor, type_id: int) -> torch.Tensor:
        # x shape: (B, L, D)
        type_emb = self.type_embedding(torch.tensor(type_id, device=x.device))
        # Add the type embedding to all tokens in the sequence
        # Unsqueeze to make it broadcastable: (D,) -> (1, 1, D)
        return x + type_emb.unsqueeze(0).unsqueeze(0)

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


   
class GoalGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        tokenizer, #add
        text_encoder,#add
        image_size,
        *,
        channels=3,
        timesteps = 1000,
        sampling_timesteps = 100,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        cond_drop_chance = 0.1 # <<<< ADD THIS
    ):
        super().__init__()
        # assert not (type(self) == GoalGaussianDiffusion and model.channels != model.out_dim)
        # assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        #add
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.cond_drop_chance = cond_drop_chance

        self.channels = channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.cond_drop_chance = cond_drop_chance # <<<< STORE THIS


        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        # IMPORTANT: Conditioning modules will be initialized later
        #self.distance_encoder = None
        #self.type_embedder = None
        #self.final_norm = None
        #self._is_cond_module_initialized = False
        

        assert self.text_encoder is not None, "text_encoder must be set before using the model"
        
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        text_embed_dim = self.text_encoder.config.hidden_size

        self.distance_encoder = nn.Sequential(
            nn.Linear(9, 256),
            nn.GELU(),
            nn.Linear(256, text_embed_dim)
        ).to(self.device)

        self.type_embedder = TypeEmbedder(num_types=2, embedding_dim=text_embed_dim).to(self.device)
        self.final_norm = nn.LayerNorm(text_embed_dim).to(self.device)
        
        self._is_cond_module_initialized = True


        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        """
        Returns the device the model is on. It's robustly determined by checking
        the device of one of its registered buffers.
        """
        return next(self.model.parameters()).device
        #return self.betas.device
             
    def _initialize_cond_modules(self):
        """
        Initializes conditioning modules once the text_encoder is available.
        This is a robust way to handle dependencies.
        """
        if self._is_cond_module_initialized:
            return

        assert self.text_encoder is not None, "text_encoder must be set before using the model"
        
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        text_embed_dim = self.text_encoder.config.hidden_size

        #这里的MLP要换成3层的那种
        self.distance_encoder = nn.Sequential(
            # Layer 1: Input(9) -> 256
            nn.Linear(9, 256),
            nn.GELU(),
            # Layer 2: 256 -> 512
            nn.Linear(256, 512),
            nn.GELU(),
            # Layer 3: 512 -> 512 (final dimension must match text_embed_dim)
            nn.Linear(512, text_embed_dim) 
            # Note: No final activation function is common practice before a LayerNorm.
            # If you want one, you can add another nn.GELU() here.
        )

        self.type_embedder = TypeEmbedder(num_types=2, embedding_dim=text_embed_dim).to(self.device)
        self.final_norm = nn.LayerNorm(text_embed_dim).to(self.device)
        
        self._is_cond_module_initialized = True
        
# In goal_diffusion_ourdataset_task_feedback.py, inside GoalGaussianDiffusion class

    def _prepare_condition_embedding(self, all_conditions):
        print(f"DEBUG: Is distance_encoder None? {self.distance_encoder is None}")
        device = self.device
        task_text_list = all_conditions['task_text']
        distance_info_tensor = all_conditions['distance_info'].to(device)
        
        # This part should be safe now because self.tokenizer and self.text_encoder are set at init
        text_inputs = self.tokenizer(
            task_text_list, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        with torch.no_grad():
            task_embed = self.text_encoder(text_input_ids)[0]

        distance_embed = self.distance_encoder(distance_info_tensor)
        typed_task_embed = self.type_embedder(task_embed, type_id=0)
        typed_distance_embed = self.type_embedder(distance_embed, type_id=1)
        
        combined_embed = torch.cat([typed_task_embed, typed_distance_embed], dim=1)
        final_embed = self.final_norm(combined_embed)

        return final_embed

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    #def model_predictions(self, x, t, x_cond, task_embed,  clip_x_start=False, rederive_pred_noise=False, guidance_weight=0):
    def model_predictions(self, x, t, x_cond, all_conditions, clip_x_start=False,rederive_pred_noise=False,  guidance_weight=0):
        # task_embed = self.text_encoder(goal).last_hidden_state
        # ====================================================================
        # (4) Same logic as in p_losses for preparing the embedding
        # ====================================================================
        final_embed = self._prepare_condition_embedding(all_conditions)
        
        # Get the conditional model output
        cond_model_output = self.model(torch.cat([x, x_cond], dim=1), t, final_embed)

        """
        #model_output = self.model(torch.cat([x, x_cond], dim=1), t, task_embed)  # x（flatten）在通道上加上一帧条件帧  21→24
        model_output = self.model(torch.cat([x, x_cond], dim=1), t, final_embed)
            # For guidance, we need an unconditional output. Let's create a null condition.
            # This is a simplification. A better CFG would use dedicated null embeddings.
            null_conditions = {
                'task_text': ["" for _ in x.shape[0]], # Empty strings for null text
                'distance_info': torch.zeros_like(all_conditions['distance_info'])
            }
            uncond_final_embed = self._prepare_condition_embedding(null_conditions)
            uncond_model_output = self.model(torch.cat([x, x_cond], dim=1), t, uncond_final_embed)

            model_output = (1 + guidance_weight) * model_output - guidance_weight * uncond_model_output
        else:
            model_output = self.model(torch.cat([x, x_cond], dim=1), t, final_embed)
        """


        if guidance_weight > 0.0:
            # Prepare the unconditional embedding
            null_conditions = {
                'task_text': ["" for _ in x.shape[0]],
                'distance_info': torch.zeros_like(all_conditions['distance_info'])
            }
            uncond_final_embed = self._prepare_condition_embedding(null_conditions)
            uncond_model_output = self.model(torch.cat([x, x_cond], dim=1), t, uncond_final_embed)
            
            # Combine conditional and unconditional outputs
            model_output = (1 + guidance_weight) * cond_model_output - guidance_weight * uncond_model_output
            #uncond_model_output = self.model(torch.cat([x, x_cond], dim=1), t, task_embed*0.0)
        else:
            model_output = cond_model_output
            
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)


        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond, all_conditions, clip_denoised=False, guidance_weight=0):
    #def p_mean_variance(self, x, t, x_cond, task_embed,  clip_denoised=False, guidance_weight=0):
        #preds = self.model_predictions(x, t, x_cond, task_embed, guidance_weight=guidance_weight)
        preds = self.model_predictions(x, t, x_cond, all_conditions, guidance_weight=guidance_weight)

        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_cond, all_conditions, guidance_weight=0):
    #def p_sample(self, x, t: int, x_cond, task_embed, guidance_weight=0):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        #model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, x_cond, task_embed, clip_denoised = True, guidance_weight=guidance_weight)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, x_cond, all_conditions, clip_denoised = True, guidance_weight=guidance_weight)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, all_conditions, return_all_timesteps=False, guidance_weight=0):
    #def p_sample_loop(self, shape, x_cond, task_embed, return_all_timesteps=False, guidance_weight=0):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, x_cond, all_conditions, guidance_weight=guidance_weight)
            #img, x_start = self.p_sample(img, t, x_cond, task_embed, guidance_weight=guidance_weight)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, x_cond, all_conditions, return_all_timesteps=False, guidance_weight=0):
    #def ddim_sample(self, shape, x_cond, task_embed, return_all_timesteps=False, guidance_weight=0):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            # self_cond = x_start if self.self_condition else None
            #pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, task_embed, clip_x_start = False, rederive_pred_noise = True, guidance_weight=guidance_weight)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, all_conditions, clip_x_start=False, rederive_pred_noise=True, guidance_weight=guidance_weight)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, x_cond, all_conditions, batch_size = 16, return_all_timesteps = False, guidance_weight=0):
    #def sample(self, x_cond, task_embed, batch_size = 16, return_all_timesteps = False, guidance_weight=0):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        #return sample_fn((batch_size, channels, image_size[0], image_size[1]), x_cond, task_embed,  return_all_timesteps = return_all_timesteps, guidance_weight=guidance_weight)
        return sample_fn((batch_size, channels, *image_size), x_cond, all_conditions,  return_all_timesteps = return_all_timesteps, guidance_weight=guidance_weight)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
        
    def p_losses(self, x_start, t, x_cond, all_conditions, noise=None):
    #def p_losses(self, x_start, t, x_cond, task_embed, noise=None):
        # x_cond:[c, 1, h, w], x:[c, 7, h, w]
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        final_embed = self._prepare_condition_embedding(all_conditions)

        #  Implement classifier-free guidance dropout if you need it.
        # This is where you would apply it.
        # e.g., keep_mask = torch.rand((b, 1, 1), device=device) > self.cond_drop_chance
        # final_embed = final_embed * keep_mask

        # Pass the final combined embedding to the UNet
        # The UNet's `task_embed` argument now receives our fused embedding.

        # Apply classifier-free guidance dropout here
        # Assuming cond_drop_chance is accessible, e.g., self.cond_drop_chance
        # You'll need to pass this value in __init__
        if self.training:
             keep_mask = (torch.rand((b, 1, 1), device=self.device) > self.cond_drop_chance).float()
             final_embed = final_embed * keep_mask

        #model_out = self.model(torch.cat([x_noisy, x_cond], dim=1), t, final_embed)


        # predict and take gradient step

        model_out = self.model(torch.cat([x, x_cond], dim=1), t, final_embed)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    #def forward(self, img, img_cond, task_embed):
    def forward(self, img, img_cond, all_conditions):
        # x_cond:[c, 1, h, w], x:[c, 7, h, w]
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}, got({h}, {w})'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        #return self.p_losses(img, t, img_cond, task_embed)
       # Pass the dictionary directly to p_losses
        return self.p_losses(img, t, img_cond, all_conditions)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        tokenizer, 
        text_encoder, 
        train_set,
        valid_set,
        channels = 3,
        *,
        train_batch_size = 1,
        valid_batch_size = 1,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 3,
        results_folder = './results',
        amp = True,
        fp16 = True,
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048, 
        cond_drop_chance=0.1,
        wandb_project=None,
        wandb_entity=None
    ):
        super().__init__()

        self.cond_drop_chance = cond_drop_chance

        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.proc_id = ''

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.channels = channels

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters

        # assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        # 3. Setup Datasets and DataLoaders
        dl = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cpu_count() # Use available CPUs
        )
        
        valid_dl = DataLoader(
            valid_set,
            batch_size=valid_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cpu_count()
        )

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        # self.results_folder.mkdir(exist_ok = True, parents = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator


        self.model, self.opt, self.dl, self.valid_dl, self.text_encoder = \
            self.accelerator.prepare(self.model, self.opt, dl, valid_dl, text_encoder)


        self.dl = cycle(self.dl)

        # 7. Setup EMA and results folder only on the main process
        self.results_folder = Path(results_folder)
        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)
            self.results_folder.mkdir(parents=True, exist_ok=True)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}{self.proc_id}.pt'))
        return str(self.results_folder / f'model-{milestone}{self.proc_id}.pt')

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    #     return fid_value
    # def encode_batch_text(self, batch_text):
    #     batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
    #     batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
    #     return batch_text_embed

    def encode_batch_text(self, batch_text):
        all_embeddings = []

        for text in batch_text:
            tokenized = self.tokenizer(text, add_special_tokens=False)
            input_ids = tokenized['input_ids']

            chunks = [
                input_ids[i:i + 75] for i in range(0, len(input_ids), 75)
            ]

            chunk_embeddings = []
            for chunk in chunks:
                # 使用 tokenizer 的 bos_token_id 和 eos_token_id 替代 cls/eos
                bos = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
                eos = self.tokenizer.eos_token_id

                assert bos is not None, "Tokenizer has no bos_token_id or cls_token_id"
                assert eos is not None, "Tokenizer has no eos_token_id"

                chunk = [bos] + chunk + [eos]
                pad_len = 77 - len(chunk)
                chunk += [self.tokenizer.pad_token_id] * pad_len

                input_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.text_encoder(input_ids=input_tensor).last_hidden_state
                    chunk_embeddings.append(embedding)

            full_embedding = torch.cat(chunk_embeddings, dim=1)
            all_embeddings.append(full_embedding)

        max_len = max([e.shape[1] for e in all_embeddings])
        padded_embeddings = []
        for emb in all_embeddings:
            pad_len = max_len - emb.shape[1]
            if pad_len > 0:
                pad_tensor = torch.zeros((1, pad_len, emb.shape[2]), device=emb.device)
                emb = torch.cat([emb, pad_tensor], dim=1)
            padded_embeddings.append(emb)

        return torch.cat(padded_embeddings, dim=0)



    def sample(self, x_conds, batch_text, batch_size=1, guidance_weight=0):
        device = self.device
        task_embeds = self.encode_batch_text(batch_text)
        #return self.ema.ema_model.sample(x_conds.to(device), task_embeds.to(device), batch_size=batch_size, guidance_weight=guidance_weight)
        return self.ema.ema_model.sample(x_conds.to(device), task_embeds.to(device), batch_size=batch_size, guidance_weight=guidance_weight)


# FILE: goal_diffusion_ourdataset_task_feedback.py
# In class Trainer:

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        if accelerator.is_main_process:
            wandb.init(project=self.wandb_project, entity = self.wandb_entity)
        else:
            wandb.init(project=self.wandb_project, entity = self.wandb_entity, mode="disabled")

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    # Data loading is correct
                    x, x_cond, all_conditions = next(self.dl)

                    with self.accelerator.autocast():
                        # Model call is correct
                        loss = self.model(x, x_cond, all_conditions)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Corrected logging
                if accelerator.is_main_process:
                    log_data = {"loss": total_loss, "step": self.step}
                    if accelerator.scaler:
                        log_data["loss_scale"] = accelerator.scaler.get_scale()
                    wandb.log(log_data)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            
                            # ====================================================================
                            #  VVV  THIS IS THE FULLY CORRECTED VALIDATION LOGIC VVV
                            # ====================================================================
                            
                            # 1. Get a single validation batch.
                            val_batch = next(iter(self.valid_dl))
                            val_xs_flat, val_x_conds, val_all_conditions = val_batch

                            # Determine number of samples to generate (don't exceed `self.num_samples`)
                            num_samples_to_generate = min(val_xs_flat.shape[0], self.num_samples)

                            # Truncate the batch if needed
                            val_xs_flat = val_xs_flat[:num_samples_to_generate]
                            val_x_conds = val_x_conds[:num_samples_to_generate]
                            val_all_conditions = {
                                'task_text': val_all_conditions['task_text'][:num_samples_to_generate],
                                'distance_info': val_all_conditions['distance_info'][:num_samples_to_generate]
                            }
                            
                            print(f"\nGenerating {num_samples_to_generate} samples for validation at step {self.step}...")

                            with self.accelerator.autocast():
                                # 2. Sample from the model using the correct data format
                                #这里是eval的
                                pred_xs_flat = self.ema.ema_model.sample(
                                    batch_size=num_samples_to_generate,
                                    x_cond=val_x_conds,
                                    all_conditions=val_all_conditions
                                )
                                """
                                pred_xs_flat = self.ema.ema_model.sample(
                                    batch_size=num_samples_to_generate,
                                    x_cond=val_x_conds,
                                    all_conditions=val_all_conditions
                                )
                                """
                        
                        print_gpu_utilization()
                        
                        # 3. Prepare tensors for saving. The ground truth is `val_xs_flat`.
                        gt_xs_flat = val_xs_flat.to('cpu')
                        pred_xs_flat = pred_xs_flat.to('cpu')
                        x_conds = val_x_conds.to('cpu')

                        # 4. Reshape for visualization
                        num_frames = gt_xs_flat.shape[1] // self.channels
                        gt_xs = rearrange(gt_xs_flat, 'b (f c) h w -> b f c h w', c=self.channels)
                        pred_xs = rearrange(pred_xs_flat, 'b (f c) h w -> b f c h w', c=self.channels)
                        first_frame = rearrange(x_conds, 'b c h w -> b 1 c h w')
                        last_frame_gt = gt_xs[:, -1:]

                        # 5. Save image grid for WandB
                        if self.step == self.save_and_sample_every:
                            os.makedirs(str(self.results_folder / 'imgs'), exist_ok=True)
                            gt_grid_tensor = torch.cat([first_frame, last_frame_gt, gt_xs], dim=1)
                            gt_grid = rearrange(gt_grid_tensor, 'b f c h w -> (b f) c h w')
                            utils.save_image(gt_grid, str(self.results_folder / 'imgs/gt_img.png'), nrow=(num_frames + 2))
                            wandb.log({"ground_truth": wandb.Image(gt_grid, caption="Ground Truth Reference")}, step=self.step)

                        os.makedirs(str(self.results_folder / 'imgs/outputs'), exist_ok=True)
                        pred_grid_tensor = torch.cat([first_frame, last_frame_gt, pred_xs], dim=1)
                        pred_grid = rearrange(pred_grid_tensor, 'b f c h w -> (b f) c h w')
                        utils.save_image(pred_grid, str(self.results_folder / f'imgs/outputs/sample-{milestone}{self.proc_id}.png'), nrow=(num_frames + 2))
                        
                        # 6. Log video to WandB
                        import imageio # Make sure imageio is installed
                        import random
                        num_videos = min(10, pred_xs.shape[0])
                        print("[DEBUG] pred_xs.shape:", pred_xs.shape)
                        print("[DEBUG] num_videos:", num_videos)
                        indices = random.sample(range(pred_xs.shape[0]), num_videos)
                        print("[DEBUG] indices:", indices)
                        for i, idx in enumerate(indices):
                            video_tensor = pred_xs[idx]
                            video_tensor = rearrange(video_tensor, 'f c h w -> f h w c')
                            video_np = (video_tensor.cpu().numpy().clip(0, 1) * 255).astype('uint8')
                            video_path = str(self.results_folder / f'sample-{milestone}{self.proc_id}_rand{i}.mp4')
                            imageio.mimsave(video_path, video_np, fps=10)
                            wandb.log({f"eval/generated_video_rand_{i}_step_{self.step}": wandb.Video(video_path, fps=10, format="mp4")}, step=self.step)

                        # 7. Save model checkpoint and run external evaluation
                        ckpt_path = self.save(milestone)
                        #current_cuda_device = self.accelerator.device.index
                        #self.run_external_eval_feedback(milestone, self.proc_id, use_feedback_text=True, cuda_device=current_cuda_device)

                pbar.update(1)

        accelerator.print('training complete')
