from goal_diffusion import GoalGaussianDiffusion_real as GoalGaussianDiffusion, Trainer_real as Trainer
from unet import Unet_real as Unet
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import SequentialDatasetv2_real as SequentialDatasetv2, parse_feedback_plan_real as parse_feedback_plan
from torch.utils.data import Subset
import argparse
import torch
import os

os.environ["WANDB_MODE"] = "disabled"
torch.cuda.empty_cache()

def main(args):
    valid_n = 2
    sample_per_seq = 8
    target_size = (128, 128)

    if args.mode == 'inference':
        train_set = valid_set = [None] # dummy
    else:
        train_set = SequentialDatasetv2(
            sample_per_seq=sample_per_seq, 
            path="/home/real_process/dataset_several/shaver_insert", 
            target_size=target_size,
            randomcrop=True,
            verbose=True
        )
        valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
        valid_set = Subset(train_set, valid_inds)

    unet = Unet()

    pretrained_model = "/home/clip"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
    text_encoder.requires_grad_(False)
    text_encoder.eval()


    diffusion = GoalGaussianDiffusion(
        model=unet,
        tokenizer=tokenizer,          # Pass tokenizer here
        text_encoder=text_encoder,    # Pass text_encoder here
        image_size=target_size,
        channels=3*(sample_per_seq-1),
        timesteps=100,
        sampling_timesteps=args.sample_steps,
        loss_type='l2',
        objective='pred_v',
        beta_schedule = 'cosine',
        min_snr_loss_weight = True,
        cond_drop_chance=0.1 # Pass the dropout chance here
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        tokenizer=tokenizer, 
        text_encoder=text_encoder,
        train_set=train_set,
        valid_set=valid_set,
        cond_drop_chance=0.1, # Keep it here for Trainer's own logic if any
        train_lr=1e-4,
        train_num_steps =150000, #130000
        save_and_sample_every =200,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =4,
        valid_batch_size =16,
        gradient_accumulate_every = 1,
        num_samples=valid_n, 
        results_folder ='../results/real',
        fp16 =True,
        amp=True,
    )


    unwrapped_diffusion_model = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped_diffusion_model.tokenizer = trainer.tokenizer
    unwrapped_diffusion_model.text_encoder = trainer.text_encoder

    if args.checkpoint_num is not None:
        print(f"加载了ckpt")
        trainer.load(args.checkpoint_num)
    
    if args.mode == 'train':
        trainer.proc_id = args.proc_id
        trainer.train()
    else:
        from PIL import Image
        from torchvision import transforms
        import imageio
        import torch
        from os.path import splitext
        from einops import rearrange  
        import random
        
        guidance_weight = args.guidance_weight
        device = trainer.device
        

        if args.batch_mode:
            
            inference_dataset = SequentialDatasetv2(
                sample_per_seq=sample_per_seq, 
                path="/home/real_process/dataset_several/shaver_insert/episode_0000", 
                target_size=target_size,
                randomcrop=False,  
                verbose=True
            )
            
            from torch.utils.data import DataLoader
            inference_dl = DataLoader(inference_dataset, batch_size=4, shuffle=False, num_workers=0)
            
            output_dir = args.output_path if args.output_path.endswith('/') else args.output_path + '/'
            os.makedirs(output_dir, exist_ok=True)
            
            num_samples_to_generate = min(50, len(inference_dataset))  
            
            sample_count = 0
            for batch_idx, (val_xs_flat, val_x_conds, val_all_conditions) in enumerate(inference_dl):
                if sample_count >= num_samples_to_generate:
                    break
                
                current_batch_size = min(val_xs_flat.shape[0], num_samples_to_generate - sample_count)
                val_xs_flat = val_xs_flat[:current_batch_size].to(device)
                val_x_conds = val_x_conds[:current_batch_size].to(device)
                val_all_conditions = {
                    'task_text': val_all_conditions['task_text'][:current_batch_size],
                    'plan_info': {
                        'action_ids': val_all_conditions['plan_info']['action_ids'][:current_batch_size].to(device),
                        'dir_ids': val_all_conditions['plan_info']['dir_ids'][:current_batch_size].to(device),
                        'dist_scalars': val_all_conditions['plan_info']['dist_scalars'][:current_batch_size].to(device)
                    }
                }
                
                with torch.no_grad():
                    pred_xs_flat = trainer.ema.ema_model.sample(
                        batch_size=current_batch_size,
                        x_cond=val_x_conds,
                        all_conditions=val_all_conditions
                    )
                                
                gt_xs_flat = val_xs_flat.to('cpu')
                pred_xs_flat = pred_xs_flat.to('cpu')
                x_conds = val_x_conds.to('cpu')
                
                num_frames = gt_xs_flat.shape[1] // trainer.channels
                gt_xs = rearrange(gt_xs_flat, 'b (f c) h w -> b f c h w', c=trainer.channels)
                pred_xs = rearrange(pred_xs_flat, 'b (f c) h w -> b f c h w', c=trainer.channels)
                first_frame = rearrange(x_conds, 'b c h w -> b 1 c h w')
                
                print(f"pred_xs.shape: {pred_xs.shape}")
                print(f"num_frames: {num_frames}")
                
                for i in range(pred_xs.shape[0]):
                    video_tensor = pred_xs[i]
                    video_tensor = rearrange(video_tensor, 'f c h w -> f h w c')
                    video_np = (video_tensor.numpy().clip(0, 1) * 255).astype('uint8')
                    
                    output_filename = f"sample_{sample_count + i:04d}.mp4"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    imageio.mimsave(output_path, video_np, fps=10)
                
                sample_count += current_batch_size
                
            
        else:            
            text = args.text
            batch_size = 1
            
            image = Image.open(args.inference_path)
            transform = transforms.Compose([
                transforms.Resize((240, 320)),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
            ])
            x_cond = transform(image).to(device)  # Shape: (C, H, W) = (3, H, W)
            
            if args.plan_path and os.path.exists(args.plan_path):
                action_ids, dir_ids, dist_scalars = parse_feedback_plan(args.plan_path)
                action_ids = action_ids.to(device)      # Shape: (max_plan_length,) = (10,)
                dir_ids = dir_ids.to(device)            # Shape: (10,)
                dist_scalars = dist_scalars.to(device)  # Shape: (10, 1)
            else:
                action_ids = torch.zeros(10, dtype=torch.long, device=device)
                dir_ids = torch.zeros(10, dtype=torch.long, device=device)
                dist_scalars = torch.zeros(10, 1, device=device)
            
            all_conditions = {
                'task_text': [text],  
                'plan_info': {
                    'action_ids': action_ids.unsqueeze(0),  
                    'dir_ids': dir_ids.unsqueeze(0),        
                    'dist_scalars': dist_scalars.unsqueeze(0) 
                }
            }
            
            with torch.no_grad():
                pred_xs_flat = trainer.ema.ema_model.sample(
                    batch_size=batch_size,
                    x_cond=x_cond.unsqueeze(0),  
                    all_conditions=all_conditions
                )
            
            
            pred_xs_flat = pred_xs_flat.to('cpu')
            
            num_frames = pred_xs_flat.shape[1] // trainer.channels
            pred_xs = rearrange(pred_xs_flat, 'b (f c) h w -> b f c h w', c=trainer.channels)
            
            video_tensor = pred_xs[0] 
            video_tensor = rearrange(video_tensor, 'f c h w -> f h w c')
            video_np = (video_tensor.numpy().clip(0, 1) * 255).astype('uint8')
            
            imageio.mimsave(args.output_path, video_np, fps=10)
            print(f'Generated {args.output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'inference']) # set to 'inference' to generate samples
    parser.add_argument('-c', '--checkpoint_num', type=int, default=None) # set to checkpoint number to resume training or generate samples
    parser.add_argument('-p', '--inference_path', type=str, default=None) # set to path to generate samples
    parser.add_argument('-t', '--text', type=str, default=None) # set to text to generate samples
    parser.add_argument('-n', '--sample_steps', type=int, default=100) # set to number of steps to sample
    parser.add_argument('-g', '--guidance_weight', type=int, default=0) # set to positive to use guidance
    parser.add_argument('-proc_id', type=str, default='2000', help='Unique identifier for this process (e.g., GPU ID or job ID)')  # new parameter
    parser.add_argument('-output_path', type=str)  # new parameter
    parser.add_argument('-plan', '--plan_path', type=str, default=None) # set to path to plan file for inference
    parser.add_argument('-batch', '--batch_mode', action='store_true', help='Enable batch inference mode')  # new parameter
    parser.add_argument('-batch_file', '--batch_file', type=str, default=None, help='Path to batch configuration file')  # new parameter
    args = parser.parse_args()
    if args.mode == 'inference':
        assert args.checkpoint_num is not None
        if not args.batch_mode:
            assert args.inference_path is not None
            assert args.text is not None
        assert args.sample_steps <= 100
    main(args)