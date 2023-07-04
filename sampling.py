from module import Model
import math
import numpy as np
from tqdm.auto import trange, tqdm

def append_zero(x):
    return np.append(x, 0)

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.):
    ramp = np.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)

def get_sigmas_exponential(n, sigma_min, sigma_max):
    """Constructs an exponential noise schedule."""
    sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), n))
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1.):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = np.linspace(1, 0, n) ** rho
    sigmas = np.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3):
    """Constructs a continuous VP noise schedule."""
    t = np.linspace(1, eps_s, n)
    sigmas = np.sqrt(np.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
    return append_zero(sigmas)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    return lambda sigma, sigma_next: np.random.randn(*x.shape).astype(np.float16)

class Scheduler:
    def __init__(self, alphas_cumprod: np.ndarray, quantize, unet_ait_exe: Model):
        super().__init__()
        self.alphas_cumprod = alphas_cumprod
        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.log_sigmas = np.log(self.sigmas)
        self.quantize = quantize
        self.sigma_data = 1.
        self.exe_module = unet_ait_exe

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_sigmas(self, n=None, denoise=1.):
        if n is None:
            return np.append(np.flip(self.sigmas, axis=0), 0)
        t_max = len(self.sigmas) - 1
        t = np.linspace(start=t_max, stop=0, num=int(n/denoise), dtype=self.sigmas.dtype)
        t = t[-n:]
        out = np.append(self.t_to_sigma(t), 0)
        return out

    def sigma_to_t(self, sigma, quantize=None):
        quantize = self.quantize if quantize is None else quantize
        log_sigma = np.log(sigma)
        dists = log_sigma - self.log_sigmas[:, None]
        if quantize:
            return np.reshape(np.argmin(np.abs(dists), axis=0), sigma.shape)
        low_idx = np.clip(np.argmax(np.cumsum(np.greater_equal(dists, 0))), a_max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)

    def t_to_sigma(self, t: np.ndarray):
        t = t.astype(np.float32)
        low_idx = np.floor(t).astype(np.int64)
        high_idx = np.ceil(t).astype(np.int64)
        w = t-low_idx
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return np.exp(log_sigma)

    def get_scalings(self, sigma):
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_out, c_in

    def get_eps(self, x, timestep, cond, uncond, cond_scale, cond_concat=None, denoise_mask=None):
        c_concat  = None
        control = None
        model_options = None
        timestep_ = timestep
        if x.shape[0] == 1 and cond.shape[0] == 1:
            x = np.concatenate([x, x], axis=0)
            timestep_ = np.concatenate([timestep, timestep], axis=0)
        c_crossattn = np.concatenate([uncond, cond], axis=0)
        c = {'c_crossattn': c_crossattn, 'c_concat': c_concat, 'control': control}
        output = self.apply_model(x, timestep_, **c)
        output = np.split(output, 2, axis=0)
        uncond = output[0]
        cond = output[1]
        return uncond + (cond - uncond) * cond_scale

    def apply_model(
        self,
        x: np.ndarray,
        t: np.ndarray,
        c_crossattn = None,
        c_concat = None,
        control = None,
        transformer_options = None,
    ):
        timesteps_pt = t
        latent_model_input = x
        encoder_hidden_states = None
        down_block_residuals = None
        mid_block_residual = None
        if c_crossattn is not None:
            encoder_hidden_states = c_crossattn
        if c_concat is not None:
            latent_model_input = np.concatenate([x] + c_concat, axis=1)
        if control is not None:
            down_block_residuals = control["output"]
            mid_block_residual = control["middle"][0]
        return self.unet_inference(
            latent_model_input=latent_model_input,
            timesteps=timesteps_pt,
            encoder_hidden_states=encoder_hidden_states,
            down_block_residuals=down_block_residuals,
            mid_block_residual=mid_block_residual,
        )

    def forward(self, input: np.ndarray, sigma: np.ndarray, **kwargs):
        c_out, c_in = [append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        eps = self.get_eps(input * c_in, self.sigma_to_t(sigma), **kwargs)
        return input + eps * c_out

    def unet_inference(
        self,
        latent_model_input,
        timesteps,
        encoder_hidden_states,
        down_block_residuals,
        mid_block_residual,
        dtype=np.float16,
    ):
        batch_size = latent_model_input.shape[0]
        height, width = latent_model_input.shape[2], latent_model_input.shape[3]
        latent_model_input = latent_model_input.astype(dtype)
        latent_model_input = np.transpose(latent_model_input, (0, 2, 3, 1))
        latent_model_input = np.ascontiguousarray(latent_model_input)
        encoder_hidden_states = encoder_hidden_states.astype(dtype)
        timesteps = timesteps.astype(dtype)
        latent_model_input = self.exe_module.numpy_to_ait_data(latent_model_input)
        encoder_hidden_states = self.exe_module.numpy_to_ait_data(encoder_hidden_states)
        timesteps = self.exe_module.numpy_to_ait_data(timesteps)

        inputs = {
            "input0": latent_model_input,
            "input1": timesteps,
            "input2": encoder_hidden_states
        }
        outputs = []
        num_outputs = len(self.exe_module.get_output_name_to_index_map())
        for i in range(num_outputs):
            shape = self.exe_module.get_output_maximum_shape(i)
            shape[0] = batch_size
            shape[1] = height
            shape[2] = width
            shape = tuple(shape)
            shape = np.empty(shape, dtype=dtype)
            outputs.append(self.exe_module.numpy_to_ait_data(shape))
        out = self.exe_module.run(
            inputs=inputs,
            outputs=outputs,
        )
        out = np.concatenate([self.exe_module.ait_data_to_numpy(x) for x in list(out.values())], axis=0)
        out = np.transpose(out, (0, 3, 1, 2))
        return out


    def sample_euler_ancestral(
        self,
        x: np.ndarray,
        sigmas: np.ndarray,
        extra_args=None,
        callback=None,
        disable=None,
        eta=1.,
        s_noise=1.,
        noise_sampler=None
    ):
        """Ancestral sampling with Euler method steps."""
        extra_args = {} if extra_args is None else extra_args
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        s_in = np.ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = self.forward(x, sigmas[i] * s_in, **extra_args)
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            d = to_d(x, sigmas[i], denoised)
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
            if sigmas[i + 1] > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        return x

    def sample_dpm_2_ancestral(
        self,
        x: np.ndarray,
        sigmas: np.ndarray,
        extra_args=None,
        callback=None,
        disable=None,
        eta=1.,
        s_noise=1.,
        noise_sampler=None
    ):
        """Ancestral sampling with DPM-Solver second-order steps."""
        extra_args = {} if extra_args is None else extra_args
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        s_in = np.ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = self.forward(x, sigmas[i] * s_in, **extra_args)
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            d = to_d(x, sigmas[i], denoised)
            if sigma_down == 0:
                # Euler method
                dt = sigma_down - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver-2
                sigma_mid = np.exp(np.interp(0.5, [0, 1], [np.log(sigmas[i]), np.log(sigma_down)]))
                dt_1 = sigma_mid - sigmas[i]
                dt_2 = sigma_down - sigmas[i]
                x_2 = x + d * dt_1
                denoised_2 = self.forward(x_2, sigma_mid * s_in, **extra_args)
                d_2 = to_d(x_2, sigma_mid, denoised_2)
                x = x + d_2 * dt_2
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        return x

    def sample_dpmpp_2s_ancestral(
        self,
        x: np.ndarray,
        sigmas: np.ndarray,
        extra_args=None,
        callback=None,
        disable=None,
        eta=1.,
        s_noise=1.,
        noise_sampler=None
    ):
        """Ancestral sampling with DPM-Solver++(2S) second-order steps."""
        extra_args = {} if extra_args is None else extra_args
        noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
        s_in = np.ones([x.shape[0]])
        sigma_fn = lambda t: np.exp(np.negative(t))
        t_fn = lambda sigma: np.negative(np.log(sigma))

        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = self.forward(x, sigmas[i] * s_in, **extra_args)
            sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            if sigma_down == 0:
                # Euler method
                d = to_d(x, sigmas[i], denoised)
                dt = sigma_down - sigmas[i]
                x = x + d * dt
            else:
                # DPM-Solver++(2S)
                t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
                r = 1 / 2
                h = t_next - t
                s = t + r * h
                x_2 = (sigma_fn(s) / sigma_fn(t)) * x - np.expm1(-h * r) * denoised
                denoised_2 = self.forward(x_2, sigma_fn(s) * s_in, **extra_args)
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - np.expm1(-h) * denoised_2
            # Noise addition
            if sigmas[i + 1] > 0:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        return x



def vae_inference(
    exe_module: Model,
    vae_input: np.ndarray,
    factor: int = 8,
    device: str = "cuda",
    dtype = np.float16,
    encoder: bool = False,
    latent_channels: int = 4,
):
    batch = vae_input.shape[0]
    height, width = vae_input.shape[2], vae_input.shape[3]
    if encoder:
        height = height // factor
        width = width // factor
    else:
        height = height * factor
        width = width * factor
    vae_input = vae_input.astype(dtype)
    vae_input = vae_input.transpose((0, 2, 3, 1))
    vae_input = np.ascontiguousarray(vae_input)
    vae_input = exe_module.numpy_to_ait_data(vae_input)
    inputs = {
        "vae_input": vae_input,
    }

    if encoder:
        sample_shape = (batch, height, width, latent_channels)
        sample = np.random.randn(*sample_shape).astype(dtype)
        # sample = sample.transpose((0, 2, 3, 1))
        sample = np.ascontiguousarray(sample)
        sample = exe_module.numpy_to_ait_data(sample)
        inputs["vae_sample"] = sample
    outputs = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        shape[0] = batch
        shape[1] = height
        shape[2] = width
        shape = tuple(shape)
        shape = np.empty(shape, dtype=dtype)
        outputs.append(exe_module.numpy_to_ait_data(shape))
    out = exe_module.run(
        inputs=inputs,
        outputs=outputs,
    )
    out = np.concatenate([exe_module.ait_data_to_numpy(x) for x in list(out.values())], axis=0)
    out = np.transpose(out, (0, 3, 1, 2))
    return out


def clip_inference(
    exe_module: Model,
    input_ids: np.ndarray,
    seqlen: int = 77,
    dtype = np.float16,
):
    batch = input_ids.shape[0]
    position_ids = np.arange(seqlen).reshape((batch, -1))
    input_ids = input_ids.astype(np.int64)
    position_ids = position_ids.astype(np.int64)
    input_ids = exe_module.numpy_to_ait_data(input_ids)
    position_ids = exe_module.numpy_to_ait_data(position_ids)
    inputs = {
        "input0": input_ids,
        "input1": position_ids,
    }
    outputs = []
    num_outputs = len(exe_module.get_output_name_to_index_map())
    for i in range(num_outputs):
        shape = exe_module.get_output_maximum_shape(i)
        shape[0] = batch
        shape = tuple(shape)
        shape = np.empty(shape, dtype=dtype)
        outputs.append(exe_module.numpy_to_ait_data(shape))
    out = exe_module.run(
        inputs=inputs,
        outputs=outputs,
    )
    out = np.concatenate([exe_module.ait_data_to_numpy(x) for x in list(out.values())], axis=0)
    return out
