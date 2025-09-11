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


ACTION_TO_ID = {
    'move': 0, 'reach': 1, 'grasp': 2, 'push': 3, 'pull': 4,
    'strike': 5, 'slide': 6, 'adjust': 7, 'rotate': 8,
    'pad': 9  # A special token for padding shorter plans
}

NUM_ACTIONS = len(ACTION_TO_ID)

# Create a mapping from 3D direction vectors to a unique integer ID
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


def parse_feedback_plan(file_path, max_plan_length=10):
    """
    Parses the plan section from a feedback file, which may also contain other sections.
    The function specifically looks for a block starting with "=== Feedback Plan ===" or "Plan:".
    
    Args:
        file_path (str): Path to the feedback file.
        max_plan_length (int): The fixed length for padding/truncating the plan.

    Returns:
        A tuple of three tensors: (action_ids, direction_ids, distance_scalars).
    """
    action_ids, direction_ids, distance_scalars = [], [], []
    
    # Define default padded values for cases where no plan is found or the file is missing
    default_actions = [ACTION_TO_ID['pad']] * max_plan_length
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
    # Strategy 1: Look for the specific "=== Feedback Plan ===" separator
    if "=== Feedback Plan ===" in content:
        # Split by the separator and take the second part
        parts = content.split("=== Feedback Plan ===")
        if len(parts) > 1:
            plan_str = parts[1]
            # Further split by "Plan:" if it exists within this block
            if "Plan:" in plan_str:
                plan_str = plan_str.split("Plan:")[1]
    
    # Strategy 2: If the separator is not found, fall back to just looking for "Plan:"
    elif "Plan:" in content:
        plan_str = content.split("Plan:")[1]
        
    plan_str = plan_str.strip()

    # If no plan content was found, return the padded defaults
    if not plan_str:
        return default_return
        
    # Regex to match each line of the plan, e.g., "1. verb [dx, dy, dz] [distance]"
    line_regex = re.compile(r"^\d+\.\s*(\w+)\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*\[\s*([\d.]+)\s*\]")
    
    for line in plan_str.split('\n'):
        line = line.strip()
        match = line_regex.match(line)
        
        if match:
            action, dx, dy, dz, distance = match.groups()
            
            # 1. Action to ID (case-insensitive)
            action_ids.append(ACTION_TO_ID.get(action.lower(), ACTION_TO_ID['pad']))
            
            # 2. Direction vector to ID
            try:
                direction_vector = (int(dx), int(dy), int(dz))
                direction_ids.append(DIR_VECTOR_TO_ID.get(direction_vector, DIR_VECTOR_TO_ID[(0,0,0)]))
            except (ValueError, KeyError):
                direction_ids.append(DIR_VECTOR_TO_ID[(0,0,0)])

            # 3. Distance scalar
            try:
                distance_scalars.append(float(distance))
            except ValueError:
                distance_scalars.append(0.0)

    # Pad or truncate the parsed plan to the fixed length
    current_length = len(action_ids)
    if current_length < max_plan_length:
        pad_length = max_plan_length - current_length
        action_ids.extend([ACTION_TO_ID['pad']] * pad_length)
        direction_ids.extend([DIR_VECTOR_TO_ID[(0,0,0)]] * pad_length)
        distance_scalars.extend([0.0] * pad_length)
    elif current_length > max_plan_length:
        action_ids = action_ids[:max_plan_length]
        direction_ids = direction_ids[:max_plan_length]
        distance_scalars = distance_scalars[:max_plan_length]

    # If parsing resulted in an empty plan (e.g., file had "Plan:" but no valid lines)
    if not action_ids:
         return default_return
               
    return torch.tensor(action_ids, dtype=torch.long), \
           torch.tensor(direction_ids, dtype=torch.long), \
           torch.tensor(distance_scalars, dtype=torch.float32).unsqueeze(-1)

def parse_feedback_plan1(file_path, max_plan_length=10):
    """
    Parses the 'Plan:' section from a feedback.txt file and converts it into tensors.
    
    Args:
        file_path (str): Path to the feedback.txt file.
        max_plan_length (int): The fixed length to which all plans will be padded or truncated.

    Returns:
        A tuple of three tensors:
        - action_ids (torch.LongTensor): Shape (max_plan_length,)
        - direction_ids (torch.LongTensor): Shape (max_plan_length,)
        - distance_scalars (torch.FloatTensor): Shape (max_plan_length, 1)
    """
    action_ids = []
    direction_ids = []
    distance_scalars = []
    
    # Default return value for missing files or empty plans
    #这里应该没有pad
    default_actions = [ACTION_TO_ID['move']] * max_plan_length
    default_directions = [DIR_VECTOR_TO_ID[(0, 0, 0)]] * max_plan_length
    default_distances = [0.0] * max_plan_length

    if not os.path.exists(file_path):
        return torch.tensor(default_actions, dtype=torch.long), \
               torch.tensor(default_directions, dtype=torch.long), \
               torch.tensor(default_distances, dtype=torch.float32).unsqueeze(-1)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract content after "Plan:" (case-insensitive)
    plan_match = re.search(r"Plan:\s*([\s\S]*)", content, re.IGNORECASE)
    plan_str = plan_match.group(1).strip() if plan_match else ""
        
    for line in plan_str.split('\n'):
        line = line.strip()
        # Regex to match "1. verb [dx, dy, dz] [distance]"
        match = re.match(r"\d+\.\s*(\w+)\s*\[\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*\]\s*\[\s*([\d.]+)\s*\]", line)
        if match:
            action, dx, dy, dz, distance = match.groups()
            
            action_ids.append(ACTION_TO_ID.get(action.lower(), ACTION_TO_ID['pad']))
            
            try:
                direction_vector = (int(dx), int(dy), int(dz))
                direction_ids.append(DIR_VECTOR_TO_ID.get(direction_vector, DIR_VECTOR_TO_ID[(0,0,0)]))
            except (ValueError, KeyError):
                direction_ids.append(DIR_VECTOR_TO_ID[(0,0,0)])

            try:
                distance_scalars.append(float(distance))
            except ValueError:
                distance_scalars.append(0.0)

    # Pad or truncate to max_plan_length
    current_length = len(action_ids)
    if current_length < max_plan_length:
        pad_length = max_plan_length - current_length
        action_ids.extend([ACTION_TO_ID['pad']] * pad_length)
        direction_ids.extend([DIR_VECTOR_TO_ID[(0,0,0)]] * pad_length)
        distance_scalars.extend([0.0] * pad_length)
    elif current_length > max_plan_length:
        action_ids = action_ids[:max_plan_length]
        direction_ids = direction_ids[:max_plan_length]
        distance_scalars = distance_scalars[:max_plan_length]

    # If no valid lines were found, return the default padded tensors
    if not action_ids:
         return torch.tensor(default_actions, dtype=torch.long), \
               torch.tensor(default_directions, dtype=torch.long), \
               torch.tensor(default_distances, dtype=torch.float32).unsqueeze(-1)
               
    return torch.tensor(action_ids, dtype=torch.long), \
           torch.tensor(direction_ids, dtype=torch.long), \
           torch.tensor(distance_scalars, dtype=torch.float32).unsqueeze(-1)

### Sequential Datasets: given first frame, predict all the future frames

class SequentialDatasetNp(Dataset):
    def __init__(self, path="../datasets/numpy/bridge_data_v1/berkeley", sample_per_seq=7, debug=False, target_size=(128, 128)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(os.path.join(path, "**/out.npy"), recursive=True)
        if debug:
            sequence_dirs = sequence_dirs[:10]
        self.sequences = []
        self.tasks = []
    
        obss, tasks = [], []
        for seq_dir in tqdm(sequence_dirs):
            obs, task = self.extract_seq(seq_dir)
            tasks.extend(task)
            obss.extend(obs)

        self.sequences = obss
        self.tasks = tasks
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("training_samples: ", len(self.sequences))
        print("Done")

    def extract_seq(self, seqs_path):
        seqs = np.load(seqs_path, allow_pickle=True)
        task = seqs_path.split('/')[-3].replace('_', ' ')
        outputs = []
        for seq in seqs:
            observations = seq["observations"]
            viewpoints = [v for v in observations[0].keys() if "image" in v]
            N = len(observations)
            for viewpoint in viewpoints:
                full_obs = [observations[i][viewpoint] for i in range(N)]
                sampled_obs = self.get_samples(full_obs)
                outputs.append(sampled_obs)
        return outputs, [task] * len(outputs)

    def get_samples(self, seq):
        N = len(seq)
        samples = []
        for i in range(self.sample_per_seq-1):
            samples.append(int(i*(N-1)/(self.sample_per_seq-1)))
        samples.append(N-1)
        return [seq[i] for i in samples]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        images = [self.transform(Image.fromarray(s)) for s in samples]
        x_cond = images[0] # first frame
        x = torch.cat(images[1:], dim=0) # all other frames
        task = self.tasks[idx]
        return x, x_cond, task
        
class SequentialDataset(SequentialDatasetNp):
    def __init__(self, path="../datasets/frederik/berkeley", sample_per_seq=7, target_size=(128, 128)):
        print("Preparing dataset...")
        sequence_dirs = get_paths(path)
        self.sequences = []
        self.tasks = []
        for seq_dir in tqdm(sequence_dirs):
            seq = self.get_samples(get_paths_from_dir(seq_dir))
            if len(seq) > 1:
                self.sequences.append(seq)
            task = seq_dir.split('/')[-6].replace('_', ' ')
            self.tasks.append(task)
        self.sample_per_seq = sample_per_seq
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        images = [self.transform(Image.open(s)) for s in samples]
        x_cond = images[0] # first frame
        x = torch.cat(images[1:], dim=0) # all other frames
        task = self.tasks[idx]
        return x, x_cond, task

class SequentialDatasetVal(SequentialDataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128)):
        print("Preparing dataset...")
        sequence_dirs = sorted([d for d in os.listdir(path) if "json" not in d], key=lambda x: int(x))
        self.sample_per_seq = sample_per_seq
        self.sequences = []
        self.tasks = []
        for seq_dir in tqdm(sequence_dirs):
            seq = self.get_samples(get_paths_from_dir(os.path.join(path, seq_dir)))
            if len(seq) > 1:
                self.sequences.append(seq)
            
        with open(os.path.join(path, "valid_tasks.json"), "r") as f:
            self.tasks = json.load(f)
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")

### Markovian datasets: given current frame, predict the next frame
class MarkovianDatasetNp(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        ### random sample 2 consecutive frames
        start_ind = np.random.randint(0, len(samples)-1)
        x_cond = torch.FloatTensor(samples[start_ind].transpose(2, 0, 1) / 255.0)
        x = torch.FloatTensor(samples[start_ind+1].transpose(2, 0, 1) / 255.0)
        task = self.tasks[idx]
        return x, x_cond, task
    
    def get_first_frame(self, idx):
        samples = self.sequences[idx]
        return torch.FloatTensor(samples[0].transpose(2, 0, 1) / 255.0)
    
class MarkovianDatasetVal(SequentialDatasetVal):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        ### random sample 2 consecutive frames
        start_ind = np.random.randint(0, len(samples)-1)
        x_cond = self.transform(Image.open(samples[start_ind]))
        x = self.transform(Image.open(samples[start_ind+1]))
        task = self.tasks[idx]
        return x, x_cond, task
    
    def get_first_frame(self, idx):
        samples = self.sequences[idx]
        return torch.FloatTensor(Image.open(samples[0]))
        
class AutoregDatasetNp(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        pred_idx = np.random.randint(1, len(samples))
        images = [torch.FloatTensor(s.transpose(2, 0, 1) / 255.0) for s in samples]
        x_cond = torch.cat(images[:-1], dim=0)
        x_cond[:, 3*pred_idx:] = 0.0
        x = images[pred_idx]
        task = self.tasks[idx]
        return x, x_cond, task
        
class AutoregDatasetNpL(SequentialDatasetNp):
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        N = len(samples)
        h, w, c = samples[0].shape
        pred_idx = np.random.randint(1, N)
        images = [torch.FloatTensor(s.transpose(2, 0, 1) / 255.0) for s in samples]
        x_cond = torch.zeros((N-1)*c, h, w)
        x_cond[(N-pred_idx-1)*3:] = torch.cat(images[:pred_idx])
        x = images[pred_idx]
        task = self.tasks[idx]
        return x, x_cond, task
    
# SSR datasets
class SSRDatasetNp(SequentialDatasetNp):
    def __init__(self, path="../datasets/numpy/bridge_data_v1/berkeley", sample_per_seq=7, debug=False, target_size=(128, 128), in_size=(48, 64), cond_noise=0.2):
        super().__init__(path, sample_per_seq, debug, target_size)
        self.downsample_tfm = T.Compose([
            T.Resize(in_size),
            T.Resize(target_size),
            T.ToTensor()
        ])

    def __getitem__(self, idx):
        samples = self.sequences[idx]
        x = torch.cat([self.transform(Image.fromarray(s)) for s in samples][1:], dim=0)
        x_cond = torch.cat([self.downsample_tfm(Image.fromarray(s)) for s in samples][1:], dim=0)
        ### apply noise on x_cond
        cond_noise = torch.randn_like(x_cond) * 0.2
        x_cond = x_cond + cond_noise
        task = self.tasks[idx]
        return x, x_cond, task
    
class SSRDatasetVal(SequentialDatasetVal):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), in_size=(48, 64)):
        print("Preparing dataset...")
        super().__init__(path, sample_per_seq, target_size)
        self.downsample_tfm = T.Compose([
            T.Resize(in_size),
            T.Resize(target_size),
            T.ToTensor()
        ])
    def __getitem__(self, idx):
        samples = self.sequences[idx]
        # images = [torch.FloatTensor(np.array(Image.open(s))[::4, ::4].transpose(2, 0, 1) / 255.0) for s in samples]
        x = torch.cat([self.transform(Image.open(s)) for s in samples][1:], dim=0)
        x_cond = torch.cat([self.downsample_tfm(Image.open(s)) for s in samples][1:], dim=0)
        ### apply noise on x_cond
        cond_noise = torch.randn_like(x_cond) * 0.2
        x_cond = x_cond + cond_noise
        task = self.tasks[idx]
        return x, x_cond, task
    
class MySeqDatasetMW(SequentialDataset):
    def __init__(self, path="../datasets/dataset_0513", sample_per_seq=8, target_size=(64, 64)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            seq = self.get_samples(sorted(glob(f"{seq_dir}*")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-3].replace("-", " "))
        
        
        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")

def parse_distance_txt(file_path):
    final_vector = torch.zeros(9, dtype=torch.float32)

    if not os.path.exists(file_path):
        return final_vector

    with open(file_path, 'r') as f:
        lines = f.readlines()

    gripper_xyz = None
    targets_xyz = []
    
    gripper_re = re.compile(r"Gripper ID \d+.*3D=\[([^\]]+)\]")
    target_re = re.compile(r"Target ID \d+.*3D=\[([^\]]+)\]")
    
    for line in lines:
        line = line.strip()
        
        if gripper_xyz is None:
            gripper_match = gripper_re.search(line)
            if gripper_match:
                try:
                    coords_str = gripper_match.group(1).split()
                    coords = [float(c) for c in coords_str]
                    if len(coords) == 3:
                        gripper_xyz = torch.tensor(coords, dtype=torch.float32)
                except (ValueError, IndexError):
                    continue
        
        target_match = target_re.search(line)
        if target_match:
            try:
                coords_str = target_match.group(1).split()
                coords = [float(c) for c in coords_str]
                if len(coords) == 3:
                    targets_xyz.append(torch.tensor(coords, dtype=torch.float32))
            except (ValueError, IndexError):
                continue
    
    if gripper_xyz is not None:
        final_vector[0:3] = gripper_xyz
        targets_xyz.sort(key=lambda t: torch.linalg.norm(t - gripper_xyz))
    
    if len(targets_xyz) > 0:
        final_vector[3:6] = targets_xyz[0]
    if len(targets_xyz) > 1:
        final_vector[6:9] = targets_xyz[1]
        
    return final_vector

class SequentialDatasetv2(Dataset):
    def __init__(self, path, sample_per_seq=8, target_size=(128, 128), randomcrop=False, **kwargs):
        print(f"Initializing Dataset for NPZ & distance.txt from path: {path}")
        self.sample_per_seq = sample_per_seq
        
        all_npz_files = glob(os.path.join(path, "**", "*sample_crop.npz"), recursive=True)
        
        self.npz_files = []
        self.tasks = {}
        self.conditional_file_paths = {}

        for npz_path in tqdm(all_npz_files, desc="Scanning for valid NPZ and distance.txt pairs"):
            dir_path = os.path.dirname(npz_path)
            cond_file_path = os.path.join(dir_path, "feedback.txt")
            
            if os.path.exists(cond_file_path):
                self.npz_files.append(npz_path)
                self.conditional_file_paths[npz_path] = cond_file_path
                
                task = "unknown_task" # Fallback
                task_txt_path = os.path.join(dir_path, "task.txt")
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
            raise FileNotFoundError(f"No valid pairs of (*sample_crop.npz + distance.txt) found in: {path}")
        
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


    def numerical_sort(self, value):
        """ Extract numbers for sorting filenames numerically. """
        match = re.search(r'frame_(\d+)', os.path.basename(value))
        return int(match.group(1)) if match else 0
    
    def build_video_transforms(self, target_size=(320, 240)):
        return video_transforms.Compose([
            video_transforms.Resize(size=target_size),  
            volume_transforms.ClipToTensor()                 
        ])

    def build_transforms(self, randomcrop, target_size):
        if randomcrop:
            return video_transforms.Compose([
                video_transforms.CenterCrop((160, 160)),    
                video_transforms.RandomCrop((128, 128)),    
                video_transforms.Resize(size=target_size),  
                volume_transforms.ClipToTensor()
            ])
        else:
            return video_transforms.Compose([
                video_transforms.CenterCrop((128, 128)),
                video_transforms.Resize(size=target_size),
                volume_transforms.ClipToTensor()
            ])

    def __len__(self):
        # It should return the length of the list of npz files
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

            x_cond = rgb_tensor[:, 0]  # First frame, Shape: (C, H, W)
            x = rearrange(rgb_tensor[:, 1:], "c f h w -> (f c) h w") # Shape: ((F-1)*C, H, W)

            task_text = self.tasks.get(npz_path, "unknown_task")
            cond_path = self.conditional_file_paths.get(npz_path)
            action_ids, dir_ids, dist_scalars = parse_feedback_plan(cond_path)
            
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
            # Fallback to the next sample to prevent training crash
            return self.__getitem__((idx + 1) % len(self))
        
class SequentialFlowDataset(Dataset):
    def __init__(self, path="../datasets/valid", sample_per_seq=7, target_size=(128, 128), frameskip=None, randomcrop=False):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        self.frame_skip = frameskip

        sequence_dirs = glob(f"{path}/**/metaworld_dataset/*/*/*/", recursive=True)
        self.tasks = []
        self.sequences = []
        self.flows = []
        for seq_dir in sequence_dirs:
            task = seq_dir.split("/")[-4]
            seq_id= int(seq_dir.split("/")[-2])
            # if task not in included_tasks or seq_id not in included_idx:
            #     continue
            seq = sorted(glob(f"{seq_dir}*.png"), key=lambda x: int(x.split("/")[-1].rstrip(".png")))
            flows = sorted(glob(f"{seq_dir}flow/*.npy        # return idx, images_tensor, x_cond, task, video_tensor"))
            self.sequences.append(seq)
            self.flows.append(np.array([np.load(flow) for flow in flows]))
            self.tasks.append(seq_dir.split("/")[-4].replace("-", " "))

        self.transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.Resize(target_size),
            T.ToTensor()
        ])
        
        print("Done")

    def get_samples(self, idx):
        seq = self.sequences[idx]
        return seq[0]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # try:
            s = self.get_samples(idx)
            x_cond = self.transform(Image.open(s)) # [c f h w]
            x = rearrange(torch.from_numpy(self.flows[idx]), "f w h c -> (f c) w h") / 128
            task = self.tasks[idx]
            return x, x_cond, task
        # except Exception as e:
        #     print(e)
        #     return self.__getitem__(idx + 1 % self.__len__()) 

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
        images = self.transform([Image.open(s) for s in samples]) # [c f h w]
        x_cond = images[:, 0] # first frame
        x = rearrange(images[:, 1:], "c f h w -> (f c) h w") # all other frames
        task = self.tasks[self.frameid2seqid[idx]]
        return x, x_cond, task

class MySeqDatasetReal(SequentialDataset):
    def __init__(self, path="../datasets/dataset_0606/processed_data", sample_per_seq=7, target_size=(48, 64)):
        print("Preparing dataset...")
        self.sample_per_seq = sample_per_seq

        sequence_dirs = glob(f"{path}/*/*/", recursive=True)
        print(f"found {len(sequence_dirs)} sequences")
        self.tasks = []
        self.sequences = []
        for seq_dir in sequence_dirs:
            seq = self.get_samples(sorted(glob(f"{seq_dir}*.png")))
            self.sequences.append(seq)
            self.tasks.append(seq_dir.split("/")[-3].replace("_", " "))
        
        self.transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])
        print("Done")


if __name__ == "__main__":
    dataset = SequentialNavDataset("../datasets/thor")
    x, x_cond, task = dataset[2]
    print(x.shape)
    print(x_cond.shape)
    print(task)

