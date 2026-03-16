import os
import sys
import torch
import random
import numpy as np
import librosa
from pathlib import Path
import urllib.request
from tqdm import tqdm
from accelerate import Accelerator
from transformers import T5Tokenizer, T5EncoderModel
from diffusers import DDIMScheduler
from src.models.conditioners import MaskDiT
from src.modules.autoencoder_wrapper import Autoencoder
from src.inference import inference
from src.utils import load_yaml_with_includes

MAX_SEED = np.iinfo(np.int32).max


configs = {'s3_xl': {'path': 'ckpts/s3/ezaudio_s3_xl.pt',
                     'url': 'https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/s3/ezaudio_s3_xl.pt',
                     'config': 'ckpts/ezaudio-xl.yml'},
           's3_l': {'path': 'ckpts/s3/ezaudio_s3_l.pt',
                     'url': 'https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/s3/ezaudio_s3_l.pt',
                     'config': 'ckpts/ezaudio-l.yml'},
          'vae': {'path': 'ckpts/vae/1m.pt', 
                  'url': 'https://huggingface.co/OpenSound/EzAudio/resolve/main/ckpts/vae/1m.pt'}
          }


class EzAudio:
    def __init__(self, model_name, ckpt_path=None, vae_path=None, device='cuda'):
        self.device = device
        config_name = configs[model_name]['config']
        if ckpt_path is None:
            ckpt_path = self.download_ckpt(configs[model_name])

        if vae_path is None:
            vae_path = self.download_ckpt(configs['vae'])

        (self.autoencoder, self.unet, self.tokenizer,
         self.text_encoder, self.noise_scheduler, self.params) = self.load_models(config_name, ckpt_path, vae_path, device)

    def download_ckpt(self, model_dict):
        local_path = Path(model_dict['path'])
        url = model_dict['url']
        # Create directories if they don't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not local_path.exists() and url:
            print(f"Downloading {local_path.name} …", file=sys.stderr)
            try:
                response = urllib.request.urlopen(url)
                total_size = int(response.headers.get('Content-Length', 0))
                with open(local_path, 'wb') as f, \
                     tqdm(total=total_size, unit='B', unit_scale=True,
                          desc=f"Downloading {local_path.name}",
                          file=sys.stderr) as pbar:
                    while True:
                        chunk = response.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
                print(f"Checkpoint saved: {local_path.name}", file=sys.stderr)
            except Exception as e:
                print(f"Error downloading checkpoint: {e}", file=sys.stderr)
        return local_path

    # Load model and configs
    def load_models(self, config_name, ckpt_path, vae_path, device):
        params = load_yaml_with_includes(config_name)

        # Load codec model
        print("Loading autoencoder …", file=sys.stderr)
        autoencoder = Autoencoder(ckpt_path=vae_path,
                                  model_type=params['autoencoder']['name'],
                                  quantization_first=params['autoencoder']['q_first']).to(device)
        autoencoder.eval()

        # Load text encoder
        t5_model = params['text_encoder']['model']
        print(f"Loading text encoder ({t5_model}) …", file=sys.stderr)
        tokenizer = T5Tokenizer.from_pretrained(t5_model)

        import logging, warnings, threading, time
        os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
        logging.disable(logging.WARNING)

        _done = threading.Event()
        _est_seconds = 12  # typical load time on Apple Silicon
        _steps = 100
        pbar = tqdm(total=_steps, desc="Loading T5", file=sys.stderr)

        def _poll():
            t0 = time.time()
            prev = 0
            while not _done.is_set():
                cur = min(95, int((time.time() - t0) / _est_seconds * _steps))
                if cur > prev:
                    pbar.update(cur - prev)
                    prev = cur
                _done.wait(0.3)
            pbar.update(_steps - prev)
            pbar.close()

        t = threading.Thread(target=_poll, daemon=True)
        t.start()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text_encoder = T5EncoderModel.from_pretrained(t5_model).to(device)
        _done.set()
        t.join()

        logging.disable(logging.NOTSET)
        os.environ.pop("ACCELERATE_LOG_LEVEL", None)
        text_encoder.eval()

        # Load main U-Net model
        print("Loading diffusion model …", file=sys.stderr)
        _done2 = threading.Event()
        pbar2 = tqdm(total=_steps, desc="Loading UNet", file=sys.stderr)

        def _poll2():
            t0 = time.time()
            prev = 0
            while not _done2.is_set():
                cur = min(95, int((time.time() - t0) / _est_seconds * _steps))
                if cur > prev:
                    pbar2.update(cur - prev)
                    prev = cur
                _done2.wait(0.3)
            pbar2.update(_steps - prev)
            pbar2.close()

        t2 = threading.Thread(target=_poll2, daemon=True)
        t2.start()
        unet = MaskDiT(**params['model']).to(device)
        unet.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
        unet.eval()
        _done2.set()
        t2.join()

        # if device == 'cuda':
        #     accelerator = Accelerator(mixed_precision="fp16")
        #     unet = accelerator.prepare(unet)

        # Load noise scheduler
        noise_scheduler = DDIMScheduler(**params['diff'])

        latents = torch.randn((1, 128, 128), device=device)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
        _ = noise_scheduler.add_noise(latents, noise, timesteps)

        return autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params

    def generate_audio(self, text, length=10,
                       guidance_scale=5, guidance_rescale=0.75, ddim_steps=100, eta=1,
                       random_seed=None, randomize_seed=False):
        neg_text = None
        length = length * self.params['autoencoder']['latent_sr']

        gt, gt_mask = None, None

        if text == '':
            guidance_scale = None
            print('empyt input')

        if randomize_seed:
            random_seed = random.randint(0, MAX_SEED)

        pred = inference(self.autoencoder, self.unet,
                         gt, gt_mask,
                         self.tokenizer, self.text_encoder,
                         self.params, self.noise_scheduler,
                         text, neg_text,
                         length,
                         guidance_scale, guidance_rescale,
                         ddim_steps, eta, random_seed,
                         self.device)

        pred = pred.cpu().numpy().squeeze(0).squeeze(0)
        # output_file = f"{save_path}/{text}.wav"
        # sf.write(output_file, pred, samplerate=params['autoencoder']['sr'])

        return self.params['autoencoder']['sr'], pred

    def editing_audio(self, text, boundary,
                      gt_file, mask_start, mask_length,
                      guidance_scale=3.5, guidance_rescale=0, ddim_steps=100, eta=1,
                      random_seed=None, randomize_seed=False):
        neg_text = None
        # max_length = 10

        if text == '':
            guidance_scale = None
            print('empyt input')

        mask_end = mask_start + mask_length

        # Load and preprocess ground truth audio
        gt, sr = librosa.load(gt_file, sr=self.params['autoencoder']['sr'])
        gt = gt / (np.max(np.abs(gt)) + 1e-9)

        audio_length = len(gt) / sr
        mask_start = min(mask_start, audio_length)
        if mask_end > audio_length:
            # outpadding mode
            padding = round((mask_end - audio_length)*self.params['autoencoder']['sr'])
            gt = np.pad(gt, (0, padding), 'constant')
            audio_length = len(gt) / sr

        output_audio = gt.copy()

        gt = torch.tensor(gt).unsqueeze(0).unsqueeze(1).to(self.device)
        boundary = min((mask_end - mask_start)/2, boundary)
        # print(boundary)

        # Calculate start and end indices
        start_idx = max(mask_start - boundary, 0)
        end_idx = min(mask_end + boundary, audio_length)
        # print(start_idx)
        # print(end_idx)

        mask_start -= start_idx
        mask_end -= start_idx

        gt = gt[:, :, round(start_idx*self.params['autoencoder']['sr']):round(end_idx*self.params['autoencoder']['sr'])]

        # Encode the audio to latent space
        gt_latent = self.autoencoder(audio=gt)
        B, D, L = gt_latent.shape
        length = L

        gt_mask = torch.zeros(B, D, L).to(self.device)
        latent_sr = self.params['autoencoder']['latent_sr']
        gt_mask[:, :, round(mask_start * latent_sr): round(mask_end * latent_sr)] = 1
        gt_mask = gt_mask.bool()

        if randomize_seed:
            random_seed = random.randint(0, MAX_SEED)

        # Perform inference to get the edited latent representation
        pred = inference(self.autoencoder, self.unet,
                         gt_latent, gt_mask,
                         self.tokenizer, self.text_encoder,
                         self.params, self.noise_scheduler,
                         text, neg_text,
                         length,
                         guidance_scale, guidance_rescale,
                         ddim_steps, eta, random_seed,
                         self.device)

        pred = pred.cpu().numpy().squeeze(0).squeeze(0)

        chunk_length = end_idx - start_idx
        pred = pred[:round(chunk_length*self.params['autoencoder']['sr'])]

        output_audio[round(start_idx*self.params['autoencoder']['sr']):round(end_idx*self.params['autoencoder']['sr'])] = pred

        pred = output_audio

        return self.params['autoencoder']['sr'], pred


