from torch.utils.data import Dataset
import os
from glob import glob
import torch
from utils import get_paths, get_paths_from_dir
from tqdm import tqdm
from PIL import Image
import numpy as np
import json
import torchvision.transforms as T
from torchvision import transforms
import random
from torchvideotransforms import video_transforms, volume_transforms
from einops import rearrange
import imageio
import re
import traceback

random.seed(0)

# Thor Setting - Action and Direction Mappings
THOR_ACTION_TO_ID = {
    'forward': 0, 'left': 1, 'right': 2, 'done': 3, 'pad': 4
}

# MetaWorld Setting - Action and Direction Mappings
MW_ACTION_TO_ID = {
    'move': 0, 'reach': 1, 'grasp': 2, 'push': 3, 'pull': 4,
    'strike': 5, 'slide': 6, 'adjust': 7, 'rotate': 8, 'pad': 9
}

DIR_VECTOR_TO_ID = {}
ID_TO_DIR_VECTOR = {}
count = 0
for x in [-1, 0, 1]:
    for y in [-1, 0, 1]:
        for z in [-1, 0, 1]:
            vector = (x, y, z)
            DIR_VECTOR_TO_ID[vector] = count
            ID_TO_DIR_VECTOR[count] = vector
            count += 1
NUM_DIRECTIONS = len(DIR_VECTOR_TO_ID)

# Real-World Setting - Action and Direction Mappings
REAL_ACTION_TO_ID = {
    'move': 0, 'grasp': 1, 'release': 2, 'lift': 3, 'place': 4,
    'toggle': 5, 'pad': 6
}

REAL_DIR_TO_ID = {
    'up': 0, 'down': 1, 'left': 2, 'right': 3,
    'forward': 4, 'backward': 5, 'pad': 6
}

ID_TO_DIR = {v: k for k, v in REAL_DIR_TO_ID.items()}

# Feedback Parsing Functions

def parse_feedback_plan_thor(feedback_path, max_length=10):
    """
    Parse Thor feedback file with format: action distance
    Example:
        right 90
        forward 2.0
        done 0
    """
    if feedback_path is None or not os.path.exists(feedback_path):
        return torch.full((max_length,), 4, dtype=torch.long), torch.zeros(max_length, 1)
    
    try:
        with open(feedback_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            return torch.full((max_length,), 4, dtype=torch.long), torch.zeros(max_length, 1)
        
        action_ids = []
        dist_scalars = []
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                action = parts[0]
                distance = float(parts[1])
                action_id = THOR_ACTION_TO_ID.get(action, 4)
                action_ids.append(action_id)
                dist_scalars.append(distance)
        
        if not action_ids:
            return torch.full((max_length,), 4, dtype=torch.long), torch.zeros(max_length, 1)
        
        while len(action_ids) < max_length:
            action_ids.append(4)
            dist_scalars.append(0.0)
        
        action_ids = action_ids[:max_length]
        dist_scalars = dist_scalars[:max_length]
        
        return (torch.tensor(action_ids, dtype=torch.long),
                torch.tensor(dist_scalars, dtype=torch.float32).unsqueeze(-1))
                
    except Exception as e:
        print(f"Error parsing feedback file {feedback_path}: {e}")
        return torch.full((max_length,), 4, dtype=torch.long), torch.zeros(max_length, 1)


def parse_feedback_plan_mw(file_path, max_plan_length=10):
    """
    Parse MetaWorld feedback file with structured plan format.
    Example:
        === Feedback Plan ===
        Plan:
        1. verb [dx, dy, dz] [distance]
    """
    action_ids, direction_ids, distance_scalars = [], [], []
    
    default_actions = [MW_ACTION_TO_ID['pad']] * max_plan_length
    default_directions = [DIR_VECTOR_TO_ID.get((0, 0, 0), 0)] * max_plan_length
    default_distances = [0.0] * max_plan_length
    
    default_return = (
        torch.tensor(default_actions, dtype=torch.long),
        torch.tensor(default_directions, dtype=torch.long),
        torch.tensor(default_distances, dtype=torch.float32).unsqueeze(-1)
    )

    if not os.path.exists(file_path):
        return default_return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    plan_str = ""
    if "=== Feedback Plan ===" in content:
        parts = content.split("=== Feedback Plan ===")
        if len(parts) > 1:
            plan_str = parts[1]
            if "Plan:" in plan_str:
                plan_str = plan_str.split("Plan:")[1]
    elif "Plan:" in content:
        plan_str = content.split("Plan:")[1]
        
    plan_str = plan_str.strip()

    if not plan_str:
        return default_return
        
    line_regex = re.compile(r"^\d+\.\s*(\w+)\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*\[\s*([\d.]+)\s*\]")
    
    for line in plan_str.split('\n'):
        line = line.strip()
        match = line_regex.match(line)
        
        if match:
            action, dx, dy, dz, distance = match.groups()
            
            action_ids.append(MW_ACTION_TO_ID.get(action.lower(), MW_ACTION_TO_ID['pad']))
            
            try:
                direction_vector = (int(dx), int(dy), int(dz))
                direction_ids.append(DIR_VECTOR_TO_ID.get(direction_vector, DIR_VECTOR_TO_ID[(0,0,0)]))
            except (ValueError, KeyError):
                direction_ids.append(DIR_VECTOR_TO_ID[(0,0,0)])

            try:
                distance_scalars.append(float(distance))
            except ValueError:
                distance_scalars.append(0.0)

    current_length = len(action_ids)
    if current_length < max_plan_length:
        pad_length = max_plan_length - current_length
        action_ids.extend([MW_ACTION_TO_ID['pad']] * pad_length)
        direction_ids.extend([DIR_VECTOR_TO_ID[(0,0,0)]] * pad_length)
        distance_scalars.extend([0.0] * pad_length)
    elif current_length > max_plan_length:
        action_ids = action_ids[:max_plan_length]
        direction_ids = direction_ids[:max_plan_length]
        distance_scalars = distance_scalars[:max_plan_length]

    if not action_ids:
         return default_return
               
    return torch.tensor(action_ids, dtype=torch.long), \
           torch.tensor(direction_ids, dtype=torch.long), \
           torch.tensor(distance_scalars, dtype=torch.float32).unsqueeze(-1)


def parse_feedback_plan_real(file_path, max_plan_length=10):
    """
    Parse real-world feedback file with GPT-generated plan format.
    Example:
        === GPT Step-by-Step Plan
        move left 0.15
        grasp down 0.05
    """
    action_ids, direction_ids, distance_scalars = [], [], []
    
    default_actions = [REAL_ACTION_TO_ID['pad']] * max_plan_length
    default_directions = [REAL_DIR_TO_ID['pad']] * max_plan_length
    default_distances = [0.0] * max_plan_length

    default_return = (
        torch.tensor(default_actions, dtype=torch.long),
        torch.tensor(default_directions, dtype=torch.long),
        torch.tensor(default_distances, dtype=torch.float32).unsqueeze(-1)
    )

    if not os.path.exists(file_path):
        return default_return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    plan_str = ""
    if "=== GPT Step-by-Step Plan" in content:
        parts = content.split("=== GPT Step-by-Step Plan")
        if len(parts) > 1:
            plan_str = parts[1]
    plan_str = plan_str.strip()

    if not plan_str:
        return default_return

    line_regex = re.compile(r"^\s*(\w+)\s+(\w+)\s+([\d.]+)")
    
    for line in plan_str.split('\n'):
        line = line.strip()
        match = line_regex.match(line)
        if match:
            action, direction, distance = match.groups()
            action_ids.append(REAL_ACTION_TO_ID.get(action.lower(), REAL_ACTION_TO_ID['pad']))
            direction_ids.append(REAL_DIR_TO_ID.get(direction.lower(), REAL_DIR_TO_ID['pad']))
            distance_scalars.append(float(distance))
    
    current_length = len(action_ids)
    if current_length < max_plan_length:
        pad_len = max_plan_length - current_length
        action_ids.extend([REAL_ACTION_TO_ID['pad']] * pad_len)
        direction_ids.extend([REAL_DIR_TO_ID['pad']] * pad_len)
        distance_scalars.extend([0.0] * pad_len)
    else:
        action_ids = action_ids[:max_plan_length]
        direction_ids = direction_ids[:max_plan_length]
        distance_scalars = distance_scalars[:max_plan_length]

    return torch.tensor(action_ids, dtype=torch.long), \
           torch.tensor(direction_ids, dtype=torch.long), \
           torch.tensor(distance_scalars, dtype=torch.float32).unsqueeze(-1)


# Thor Datasets

class SequentialNavDataset(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=8, target_size=(64, 64)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(f"{path}/**/thor_dataset/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-3]
            seq = sorted(glob(f"{seq_dir}frames/*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
            self.sequences.append(seq)
            self.tasks.append(task)

        self.transform = video_transforms.Compose([
            video_transforms.Resize(target_size),
            volume_transforms.ClipToTensor()
        ])

        num_seqs = len(self.sequences)
        num_frames = sum([len(seq) for seq in self.sequences])
        self.num_frames = num_frames
        self.frameid2seqid = [i for i, seq in enumerate(self.sequences) for _ in range(len(seq))]
        self.frameid2seq_subid = [f - self.frameid2seqid.index(self.frameid2seqid[f]) for f in range(num_frames)]

        print(f"Found {num_seqs} seqs, {num_frames} frames in total")
        print("Done")

    def get_samples(self, idx):
        seqid = self.frameid2seqid[idx]
        seq = self.sequences[seqid]
        start_idx = self.frameid2seq_subid[idx]
        
        samples = [i if i < len(seq) else -1 for i in range(start_idx, start_idx+self.sample_per_seq)]
        return [seq[i] for i in samples]
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        samples = self.get_samples(idx)
        images = self.transform([Image.open(s) for s in samples])
        x_cond = images[:, 0]
        x = rearrange(images[:, 1:], "c f h w -> (f c) h w")
        task = self.tasks[self.frameid2seqid[idx]]
        return x, x_cond, task


class SequentialGifDataset(Dataset):
    """Dataset class for loading data from GIF files with structured feedback"""
    def __init__(self, path="/home/datasets/thor_final", sample_per_seq=8, target_size=(64, 64)):
        print("Preparing GIF dataset...")
        self.sample_per_seq = sample_per_seq
        self.target_size = target_size
        
        gif_files = glob(f"{path}/**/*.gif", recursive=True)
        print(f"Found {len(gif_files)} GIF files")
        
        self.sequences = []
        self.tasks = []
        self.feedback_paths = []
        
        for idx, gif_path in enumerate(tqdm(gif_files)):
            path_parts = gif_path.split('/')
            task_name = path_parts[-3]
            segment_name = path_parts[-1].replace('.gif', '')
            
            dir_path = os.path.dirname(gif_path)
            if "segment" in segment_name:
                segment_num = segment_name.split("segment")[1]
                feedback_file = f"feedback_seg{segment_num}.txt"
                feedback_path = os.path.join(dir_path, feedback_file)
            else:
                feedback_path = None
            
            if idx < 5:
                print(f"GIF path: {gif_path}")
                print(f"Segment name: {segment_name}")
                print(f"Directory path: {dir_path}")
                print(f"Feedback path: {feedback_path}")
                if feedback_path and os.path.exists(feedback_path):
                    print(f"Feedback file exists: {feedback_path}")
                else:
                    print(f"Feedback file does not exist or is None")
            
            try:
                gif_frames = imageio.mimread(gif_path)
                if len(gif_frames) >= 2:
                    self.sequences.append(gif_frames)
                    self.tasks.append(task_name)
                    self.feedback_paths.append(feedback_path)
                else:
                    print(f"Warning: {gif_path} only has {len(gif_frames)} frames, need at least 2 frames")
            except Exception as e:
                print(f"Error: Cannot read {gif_path}: {e}")
        
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
        
        print(f"Dataset preparation complete, {len(self.sequences)} sequences in total")
        print("Done")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        gif_frames = self.sequences[idx]
        task = self.tasks[idx]
        feedback_path = self.feedback_paths[idx]
        
        total_frames = len(gif_frames)
        if total_frames == self.sample_per_seq:
            sampled_frames = gif_frames
        else:
            indices = [int(i * (total_frames - 1) / (self.sample_per_seq - 1)) for i in range(self.sample_per_seq)]
            sampled_frames = [gif_frames[i] for i in indices]
        
        images = []
        for frame in sampled_frames:
            if isinstance(frame, np.ndarray):
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    pil_image = Image.fromarray(frame.astype('uint8'))
                else:
                    pil_image = Image.fromarray(frame.astype('uint8')).convert('RGB')
            else:
                pil_image = frame
            images.append(self.transform(pil_image))
        
        images_tensor = torch.stack(images)
        x_cond = images_tensor[0]
        x = images_tensor[1:].flatten(0, 1)
        
        action_ids, dist_scalars = parse_feedback_plan_thor(feedback_path)
        
        all_conditions = {
            'task_text': task,
            'plan_info': {
                'action_ids': action_ids,
                'dist_scalars': dist_scalars
            }
        }
        
        return x, x_cond, all_conditions


# MetaWorld Dataset

class SequentialDatasetv2_mw(Dataset):
    def __init__(self, path, sample_per_seq=8, target_size=(128, 128), randomcrop=False, **kwargs):
        print(f"Initializing Dataset for NPZ & feedback.txt from path: {path}")
        self.sample_per_seq = sample_per_seq
        
        all_npz_files = glob(os.path.join(path, "**", "*sample_crop.npz"), recursive=True)

        self.npz_files = []
        self.tasks = {}
        self.conditional_file_paths = {}

        for npz_path in tqdm(all_npz_files, desc="Scanning for valid NPZ and feedback.txt pairs"):
            dir_path = os.path.dirname(npz_path)
            cond_file_path = os.path.join(dir_path, "feedback.txt")
            
            if os.path.exists(cond_file_path):
                self.npz_files.append(npz_path)
                self.conditional_file_paths[npz_path] = cond_file_path
                
                task = "unknown_task"
                task_txt_path = os.path.join(dir_path, "task.txt")
                task_raw = "unknown_task"
                if os.path.exists(task_txt_path):
                    with open(task_txt_path, 'r', encoding='utf-8') as f:
                        task_lines = f.readlines()
                        if task_lines:
                            task = task_lines[0].strip()
                else:
                    try:
                        parent_dir = os.path.dirname(dir_path)
                        grandparent_dir = os.path.dirname(parent_dir)
                        task_raw = os.path.basename(grandparent_dir) 
                    except IndexError:
                        task_raw = "unknown_task"
                        pass

                if "-v2-goal-observable" in task_raw:
                    task = task_raw.split("-v2-goal-observable")[0]
                else:
                    task = task_raw 
                
                self.tasks[npz_path] = task

        if not self.npz_files:
            raise FileNotFoundError(f"No valid pairs of (*sample_crop.npz + feedback.txt) found in: {path}")
        
        print(f"Found {len(self.npz_files)} valid data pairs.")

        self.transform = self._build_transforms(randomcrop, target_size)

    def _build_transforms(self, randomcrop, target_size):
        compose_list = []
        if randomcrop:
            compose_list.append(video_transforms.RandomCrop(target_size))
        else:
            compose_list.append(video_transforms.CenterCrop(target_size))
        
        compose_list.extend([
            video_transforms.Resize(target_size),
            volume_transforms.ClipToTensor(channel_nb=3)
        ])
        return video_transforms.Compose(compose_list)

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                images_np = data['images']

            if images_np.shape[0] < self.sample_per_seq:
                return self.__getitem__((idx + 1) % len(self))
            
            rgb_pil = [Image.fromarray(img) for img in images_np[:self.sample_per_seq]]
            rgb_tensor = self.transform(rgb_pil)

            x_cond = rgb_tensor[:, 0]
            x = rearrange(rgb_tensor[:, 1:], "c f h w -> (f c) h w")

            task_text = self.tasks.get(npz_path, "unknown_task")
            cond_path = self.conditional_file_paths.get(npz_path)
            action_ids, dir_ids, dist_scalars = parse_feedback_plan_mw(cond_path)
            
            all_conditions = {
                'task_text': task_text,
                'plan_info': {
                    'action_ids': action_ids,
                    'dir_ids': dir_ids,
                    'dist_scalars': dist_scalars
                }
            }
            return x, x_cond, all_conditions

        except Exception as e:
            print(f"ERROR loading data for {npz_path}: {e}")
            traceback.print_exc()
            return self.__getitem__((idx + 1) % len(self))


class SequentialDatasetv2_real(Dataset):
    def __init__(self, path, sample_per_seq=8, target_size=(128,128), randomcrop=False, verbose=False, **kwargs):
        print(f"Initializing Dataset for GIF & feedback.txt from path: {path}")
        self.sample_per_seq = sample_per_seq
        self.target_size = target_size
        self.verbose = verbose

        all_feedback_files = glob(os.path.join(path, "**", "*_feedback.txt"), recursive=True)
        self.clip_files = []
        self.feedback_files = {}
        self.tasks = {}

        for feedback_path in tqdm(all_feedback_files, desc="Scanning for feedback.txt + GIF"):
            dir_path = os.path.dirname(feedback_path)
            base_name = os.path.basename(feedback_path).replace("_feedback.txt","")
            clip_path = os.path.join(dir_path, base_name+".gif")
            if os.path.exists(clip_path):
                self.clip_files.append(clip_path)
                self.feedback_files[clip_path] = feedback_path

                task_txt_path = os.path.join(dir_path, "task.txt")
                task = "unknown_task"
                if os.path.exists(task_txt_path):
                    with open(task_txt_path,'r',encoding='utf-8') as f:
                        lines = f.readlines()
                        if lines:
                            task = lines[0].strip()
                else:
                    try:
                        parent_dir = os.path.dirname(dir_path)
                        grandparent_dir = os.path.basename(parent_dir)
                        task = grandparent_dir.split("-v2-goal-observable")[0] if "-v2-goal-observable" in grandparent_dir else grandparent_dir
                    except:
                        task = "unknown_task"
                self.tasks[clip_path] = task

        if not self.clip_files:
            raise FileNotFoundError(f"No valid feedback.txt + GIF pairs in {path}")

        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

        if self.verbose:
            print("\n" + "="*20 + " Dataset Debug Info " + "="*20)
            for i in range(min(3, len(self.clip_files))):
                try:
                    x, x_cond, all_conditions = self[i]
                    print(f"[Sample {i}]")
                    print(f"  x_cond.shape: {x_cond.shape}")
                    print(f"  x.shape: {x.shape}")
                    print(f"  task_text: {all_conditions['task_text']}")
                    plan = all_conditions['plan_info']
                    print(f"  action_ids: {plan['action_ids']}")
                    print(f"  dir_ids: {plan['dir_ids']}")
                    print(f"  dist_scalars: {plan['dist_scalars'].squeeze().tolist()}")
                except Exception as e:
                    print(f"  ERROR loading sample {i}: {e}")
            print("="*60 + "\n")

    def __len__(self):
        return len(self.clip_files)

    def __getitem__(self, idx):
        clip_path = self.clip_files[idx]
        feedback_path = self.feedback_files[clip_path]

        frames = []
        try:
            gif = Image.open(clip_path)
            while True:
                frame = gif.copy().convert("RGB")
                frames.append(self.transform(frame))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        if len(frames) < self.sample_per_seq:
            frames += [frames[-1]] * (self.sample_per_seq - len(frames))
        frames = torch.stack(frames[:self.sample_per_seq], dim=1)

        x_cond = frames[:, 0]
        x = rearrange(frames[:, 1:], "c f h w -> (f c) h w")

        action_ids, dir_ids, dist_scalars = parse_feedback_plan_real(feedback_path)
        all_conditions = {
            'task_text': [self.tasks[clip_path]],
            'plan_info': {
                'action_ids': action_ids,
                'dir_ids': dir_ids,
                'dist_scalars': dist_scalars
            }
        }

        return x, x_cond, all_conditions