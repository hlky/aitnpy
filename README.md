# aitnpy

AITemplate🤝Numpy


```
python .\numpy_test.py --help
usage: numpy_test.py [-h] [--prompt PROMPT] [--negative_prompt NEGATIVE_PROMPT] [--model_path MODEL_PATH] [--unet_module_path UNET_MODULE_PATH] [--vae_module_path VAE_MODULE_PATH] [--clip_module_path CLIP_MODULE_PATH]
                     [--output_path OUTPUT_PATH]

options:
  -h, --help            show this help message and exit
  --prompt PROMPT
  --negative_prompt NEGATIVE_PROMPT
  --model_path MODEL_PATH
  --unet_module_path UNET_MODULE_PATH
  --vae_module_path VAE_MODULE_PATH
  --clip_module_path CLIP_MODULE_PATH
  --output_path OUTPUT_PATH
```

e.g. 

```
python .\numpy_test.py --model_path "H:/models/checkpoints/v1-5-pruned-emaonly.safetensors" --unet_module_path "C:\source\sdr\v1_unet_64_1024.so" --clip_module_path "C:\source\sdr\v1_clip.so" --vae_module_path "C:\source\sdr\v1_vae_64_1024.so"
```