import numpy as np

def map_unet(pt_mod, in_channels=None, conv_in_key=None, dim=320, device="cuda", dtype="float16", arange_only=False):
    if in_channels is not None and conv_in_key is None:
        raise ValueError("conv_in_key must be specified if in_channels is not None for padding")
    pt_params = pt_mod
    params_ait = {}
    params_ait["arange"] = np.arange(start=0, stop=dim // 2, dtype=np.float32)
    if arange_only:
        return params_ait
    for key, arr in pt_params.items():
        arr: np.ndarray = arr.astype(np.float16)
        if key.startswith("model.diffusion_model."):
            key = key.replace("model.diffusion_model.", "")
        if len(arr.shape) == 4:
            arr = arr.transpose((0, 2, 3, 1))
            arr = np.ascontiguousarray(arr)
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = np.split(arr, 2, axis=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = np.split(arr, 2, axis=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr

    if conv_in_key is not None:
        if in_channels > 0 and in_channels < 4:
            pad_by = 4 - in_channels
        elif in_channels > 4 and in_channels < 8:
            pad_by = 8 - in_channels
        elif in_channels > 8 and in_channels < 12:
            pad_by = 12 - in_channels
        else:
            pad_by = 0
        
        params_ait[conv_in_key] = np.pad(params_ait[conv_in_key],((0, 0), (0, pad_by), (0, 0), (0, 0)))

    return params_ait



def map_vae(pt_module, device="cuda", dtype="float16", encoder=False):
    pt_params = pt_module
    params_ait = {}
    quant_key = "post_quant" if encoder else "quant"
    vae_key = "decoder" if encoder else "encoder"
    for key, arr in pt_params.items():
        if key.startswith(vae_key):
            continue
        if key.startswith(quant_key):
            continue
        arr: np.ndarray = arr.astype(np.float16)
        key = key.replace(".", "_")
        if (
            "conv" in key
            and "norm" not in key
            and key.endswith("_weight")
            and len(arr.shape) == 4
        ):
            arr = arr.transpose((0, 2, 3, 1))
            arr = np.ascontiguousarray(arr)
            params_ait[key] = arr
        elif key.endswith("proj_attn_weight"):
            prefix = key[: -len("proj_attn_weight")]
            key = prefix + "attention_proj_weight"
            params_ait[key] = arr
        elif key.endswith("to_out_0_weight"):
            prefix = key[: -len("to_out_0_weight")]
            key = prefix + "attention_proj_weight"
            params_ait[key] = arr
        elif key.endswith("proj_attn_bias"):
            prefix = key[: -len("proj_attn_bias")]
            key = prefix + "attention_proj_bias"
            params_ait[key] = arr
        elif key.endswith("to_out_0_bias"):
            prefix = key[: -len("to_out_0_bias")]
            key = prefix + "attention_proj_bias"
            params_ait[key] = arr
        elif key.endswith("query_weight"):
            prefix = key[: -len("query_weight")]
            key = prefix + "attention_proj_q_weight"
            params_ait[key] = arr
        elif key.endswith("to_q_weight"):
            prefix = key[: -len("to_q_weight")]
            key = prefix + "attention_proj_q_weight"
            params_ait[key] = arr
        elif key.endswith("query_bias"):
            prefix = key[: -len("query_bias")]
            key = prefix + "attention_proj_q_bias"
            params_ait[key] = arr
        elif key.endswith("to_q_bias"):
            prefix = key[: -len("to_q_bias")]
            key = prefix + "attention_proj_q_bias"
            params_ait[key] = arr
        elif key.endswith("key_weight"):
            prefix = key[: -len("key_weight")]
            key = prefix + "attention_proj_k_weight"
            params_ait[key] = arr
        elif key.endswith("key_bias"):
            prefix = key[: -len("key_bias")]
            key = prefix + "attention_proj_k_bias"
            params_ait[key] = arr
        elif key.endswith("value_weight"):
            prefix = key[: -len("value_weight")]
            key = prefix + "attention_proj_v_weight"
            params_ait[key] = arr
        elif key.endswith("value_bias"):
            prefix = key[: -len("value_bias")]
            key = prefix + "attention_proj_v_bias"
            params_ait[key] = arr
        elif key.endswith("to_k_weight"):
            prefix = key[: -len("to_k_weight")]
            key = prefix + "attention_proj_k_weight"
            params_ait[key] = arr
        elif key.endswith("to_v_weight"):
            prefix = key[: -len("to_v_weight")]
            key = prefix + "attention_proj_v_weight"
            params_ait[key] = arr
        elif key.endswith("to_k_bias"):
            prefix = key[: -len("to_k_bias")]
            key = prefix + "attention_proj_k_bias"
            params_ait[key] = arr
        elif key.endswith("to_v_bias"):
            prefix = key[: -len("to_v_bias")]
            key = prefix + "attention_proj_v_bias"
            params_ait[key] = arr
        else:
            params_ait[key] = arr
    if encoder:
        
        params_ait["encoder_conv_in_weight"] = np.pad(params_ait["encoder_conv_in_weight"], ((0, 0), (0, 0), (0, 0), (0, 1)))
    return params_ait


def map_clip(pt_mod, device="cuda", dtype="float16"):
    pt_params = pt_mod
    params_ait = {}
    for key, arr in pt_params.items():
        arr = arr.astype(np.float16)
        name = key.replace("text_model.", "")
        ait_name = name.replace(".", "_")
        if ait_name == "embeddings_position_ids":
            continue
        if name.endswith("out_proj.weight"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif name.endswith("out_proj.bias"):
            ait_name = ait_name.replace("out_proj", "proj")
        elif "q_proj" in name:
            ait_name = ait_name.replace("q_proj", "proj_q")
        elif "k_proj" in name:
            ait_name = ait_name.replace("k_proj", "proj_k")
        elif "v_proj" in name:
            ait_name = ait_name.replace("v_proj", "proj_v")
        params_ait[ait_name] = arr
    
    # params_ait["embeddings_positions_ids"] = np.arange(start=0, stop=77, dtype=np.int64)

    return params_ait
