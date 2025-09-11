from goal_diffusion_distance import GoalGaussianDiffusion, Trainer
from unet import UnetMW as Unet
from transformers import CLIPTextModel, CLIPTokenizer
from datasets_feedback import SequentialDatasetv2
from torch.utils.data import Subset
import argparse

def main(args):
    valid_n = 2
    sample_per_seq = 8
    target_size = (128, 128)

    if args.mode == 'inference':
        train_set = valid_set = [None] # dummy
    else:
        train_set = SequentialDatasetv2(
            sample_per_seq=sample_per_seq, 
            # path="../datasets/metaworld", 
            path="/villa/VideoAgent_exp/dataset/high_quality_final_1_selected/expert/fine_prob/", 
            target_size=target_size,
            randomcrop=True
        )
        valid_inds = [i for i in range(0, len(train_set), len(train_set)//valid_n)][:valid_n]
        valid_set = Subset(train_set, valid_inds)

    unet = Unet()

    pretrained_model = "/villa/VideoAgent_exp/clip"
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
        train_num_steps =100000,
        save_and_sample_every =200,
        ema_update_every = 10,
        ema_decay = 0.999,
        train_batch_size =8,
        valid_batch_size =16,
        gradient_accumulate_every = 1,
        num_samples=valid_n, 
        results_folder ='../results/mw',
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
        text = args.text
        guidance_weight = args.guidance_weight
        image = Image.open(args.inference_path)
        batch_size = 1
        transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
        image = transform(image)
        

        dummy_distance_info = torch.zeros(batch_size, 1, 9) 
        
        inference_conditions = {
            'task_text': [text] * batch_size,
            'distance_info': dummy_distance_info
        }
        
        output = trainer.sample(image.unsqueeze(0), [text], batch_size, guidance_weight).cpu()
        output = output[0].reshape(-1, 3, *target_size)
        output = torch.cat([image.unsqueeze(0), output], dim=0)
        root, ext = splitext(args.inference_path)
        # output_gif = root + '_out.gif'
        output = (output.cpu().numpy().transpose(0, 2, 3, 1).clip(0, 1) * 255).astype('uint8')
        imageio.mimsave(args.output_path, output, duration=200, loop=1000)
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
    args = parser.parse_args()
    if args.mode == 'inference':
        assert args.checkpoint_num is not None
        assert args.inference_path is not None
        assert args.text is not None
        assert args.sample_steps <= 100
    main(args)