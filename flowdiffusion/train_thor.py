from goal_diffusion import GoalGaussianDiffusion_thor as GoalGaussianDiffusion, Trainer_thor as Trainer
from unet import UnetThor as Unet
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import SequentialNavDataset, SequentialGifDataset
from torch.utils.data import Subset
import argparse
import torch
import os
from einops import rearrange, repeat
from PIL import Image
from torchvision import transforms
import imageio
from os.path import splitext

os.environ["WANDB_MODE"] = "disabled"
torch.cuda.empty_cache()


def main(args):
    valid_n = 1
    sample_per_seq = 8
    target_size = (64, 64)

    if args.mode == 'inference':
        train_set = valid_set = [None]
    else:
        train_set = SequentialGifDataset(
            sample_per_seq=sample_per_seq,
            path="/home/datasets/thor",
            target_size=target_size,
        )

        valid_inds = [i for i in range(0, len(train_set), max(1, len(train_set) // valid_n))][:valid_n]
        valid_set = Subset(train_set, valid_inds)


    unet = Unet()  

    pretrained_model = "/DATA/clip"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_encoder = text_encoder.to(device)

    diffusion = GoalGaussianDiffusion(
        channels=3 * (sample_per_seq - 1),
        model=unet,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        image_size=target_size,
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule='cosine',
        min_snr_loss_weight=True,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        train_set=train_set,
        valid_set=valid_set,
        train_lr=1e-4,
        train_num_steps=80000,
        save_and_sample_every=200,
        ema_update_every=10,
        ema_decay=0.999,
        train_batch_size=4,
        valid_batch_size=32,
        gradient_accumulate_every=1,
        num_samples=valid_n,
        results_folder='/home/results/thor',
        fp16=True,
        amp=True,
        wandb_project=None,
        wandb_entity=None,
    )

    if args.checkpoint_num is not None:
        trainer.load(args.checkpoint_num)

    if args.mode == 'train':
        trainer.train()
    else:
        text = args.text
        image = Image.open(args.inference_path).convert("RGB")
        batch_size = 1
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
        image = transform(image)

        output = trainer.sample(image.unsqueeze(0), [text], batch_size).cpu()
        output = output[0].reshape(-1, 3, *target_size)
        output = torch.cat([image.unsqueeze(0), output], dim=0)

        root, ext = splitext(args.inference_path)
        output_gif = root + '_out.gif'
        output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')
        imageio.mimsave(output_gif, output, duration=200, loop=1000)
        print(f'Generated {output_gif}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'inference'])
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None)
    parser.add_argument('-p', '--inference_path', type=str, default=None)
    parser.add_argument('-t', '--text', type=str, default=None)
    parser.add_argument('-n', '--sample_steps', type=int, default=100)
    parser.add_argument('-g', '--guidance_weight', type=int, default=0)
    parser.add_argument('-proc_id', type=str, default='3001', help='Process ID for logging')
    args = parser.parse_args()

    if args.mode == 'inference':
        assert args.checkpoint_num is not None, 
        assert args.inference_path is not None, 
        assert args.text is not None, 
        assert args.sample_steps <= 100, 

    main(args)
