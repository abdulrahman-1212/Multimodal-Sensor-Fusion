from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
from transFuser import Config


config = Config()

## 1. Enhanced KITTI Data Loader
class KITTIDataset(Dataset):
    def __init__(self, root_dir, samples, transform=None, is_train=True):
        self.root_dir = root_dir
        self.samples = samples
        self.transform = transform
        self.is_train = is_train
        self.calib_dir = os.path.join(root_dir, 'calib')
        self.image_dir = os.path.join(root_dir, 'image_2')
        self.lidar_dir = os.path.join(root_dir, 'velodyne')
        self.label_dir = os.path.join(root_dir, 'label_2')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, f'{sample_name}.png')
        image = Image.open(img_path).convert('RGB')
        image = image.resize(config.image_size)
        
        # Load point cloud
        lidar_path = os.path.join(self.lidar_dir, f'{sample_name}.bin')
        point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        
        # Load calibration
        calib = self.load_calibration(os.path.join(self.calib_dir, f'{sample_name}.txt'))
        
        # Load labels
        label_path = os.path.join(self.label_dir, f'{sample_name}.txt')
        labels = self.parse_labels(label_path)
        
        # Convert point cloud to camera coordinates
        point_cloud_cam = self.lidar_to_cam(point_cloud, calib)
        
        # Apply transforms
        if self.transform:
            image, point_cloud_cam = self.transform(image, point_cloud_cam)
        
        # Convert to tensors
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        point_cloud_cam = torch.from_numpy(point_cloud_cam).float()
        
        # Prepare targets
        target = self.prepare_targets(labels)
        
        return {
            'image': image,
            'point_cloud': point_cloud_cam,
            'target': target,
            'sample_name': sample_name
        }
    
    def load_calibration(self, calib_path):
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key.strip()] = np.array([float(x) for x in value.strip().split()])
        return calib
    
    def lidar_to_cam(self, points, calib):
        # Convert LiDAR points to camera coordinates
        R = calib['R0_rect'].reshape(3, 3)
        Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
        
        # Homogeneous coordinates
        points_hom = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
        
        # Transform to camera coordinates
        points_cam = (R @ Tr_velo_to_cam @ points_hom.T).T
        
        # Keep reflectance if available
        if points.shape[1] > 3:
            points_cam = np.hstack([points_cam, points[:, 3:]])
        
        return points_cam
    
    def parse_labels(self, label_path):
        objects = []
        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) < 15:
                    continue
                
                obj = {
                    'type': values[0],
                    'truncated': float(values[1]),
                    'occluded': int(values[2]),
                    'alpha': float(values[3]),
                    'bbox': [float(x) for x in values[4:8]],
                    'dimensions': [float(x) for x in values[8:11]],
                    'location': [float(x) for x in values[11:14]],
                    'rotation_y': float(values[14]),
                    'score': float(values[15]) if len(values) > 15 else 1.0
                }
                objects.append(obj)
        return objects
    
    def prepare_targets(self, labels):
        # Convert KITTI labels to training targets
        targets = []
        
        for obj in labels:
            if obj['type'].lower() not in ['car', 'pedestrian', 'cyclist']:
                continue
                
            # Class mapping
            class_id = {'car': 0, 'pedestrian': 1, 'cyclist': 2}[obj['type'].lower()]
            
            target = {
                'class_id': class_id,
                'location': obj['location'],
                'dimensions': obj['dimensions'],
                'rotation_y': obj['rotation_y'],
                'truncated': obj['truncated'],
                'occluded': obj['occluded']
            }
            targets.append(target)
        
        return targets

## 2. Enhanced Data Transformations
class KITTITransform:
    def __init__(self, augment=True):
        self.augment = augment
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])
        
    def __call__(self, image, point_cloud):
        # Convert PIL Image to numpy array
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Random horizontal flip
        if self.augment and np.random.rand() > 0.5:
            image = np.fliplr(image)
            point_cloud[:, 1] = -point_cloud[:, 1]  # Flip y-coordinate
            
        # Normalize image
        image = (image - self.image_mean) / self.image_std
        
        # Process point cloud
        point_cloud = self.filter_point_cloud(point_cloud)
        
        return image, point_cloud
    
    def filter_point_cloud(self, points):
        # Filter points in front of camera and within reasonable range
        mask = (points[:, 2] > 0) & (points[:, 2] < 80) & \
               (np.abs(points[:, 0]) < 40) & (np.abs(points[:, 1]) < 40)
        points = points[mask]
        
        # Random sampling if too many points
        if len(points) > config.max_points:
            idx = np.random.choice(len(points), config.max_points, replace=False)
            points = points[idx]
        
        # Normalize point cloud
        points[:, :3] = (points[:, :3] - np.mean(points[:, :3], axis=0)) / \
                       (np.std(points[:, :3], axis=0) + 1e-6)
        
        return points
    
## 4. Training Utilities
class KITTICollator:
    def __init__(self):
        pass
    
    def __call__(self, batch):
        images = torch.stack([item['image'] for item in batch])
        
        # Pad point clouds to have same number of points
        max_points = max(item['point_cloud'].shape[0] for item in batch)
        padded_pcs = torch.zeros(len(batch), max_points, 3)
        
        for i, item in enumerate(batch):
            pc = item['point_cloud']
            padded_pcs[i, :pc.shape[0]] = pc[:, :3]  # Only use xyz
        
        # Prepare targets
        max_objects = max(len(item['target']) for item in batch)
        padded_targets = torch.zeros(len(batch), max_objects, 7)
        class_labels = torch.zeros(len(batch), max_objects, dtype=torch.long)
        
        for i, item in enumerate(batch):
            targets = item['target']
            for j, target in enumerate(targets[:max_objects]):
                padded_targets[i, j] = torch.tensor([
                    target['location'][0],
                    target['location'][1],
                    target['location'][2],
                    target['dimensions'][0],
                    target['dimensions'][1],
                    target['dimensions'][2],
                    target['rotation_y']
                ])
                class_labels[i, j] = target['class_id']
        
        return {
            'image': images,
            'point_cloud': padded_pcs,
            'target': padded_targets,
            'class_label': class_labels,
            'sample_name': [item['sample_name'] for item in batch]
        }

class KITTILoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, preds, targets):
        reg_pred, cls_pred = preds
        reg_target, cls_target = targets
        
        # Regression loss (only for positive classes)
        reg_mask = (cls_target != 0).float()  # 0 is background
        reg_loss = self.reg_loss(reg_pred, reg_target) * reg_mask.unsqueeze(-1)
        reg_loss = reg_loss.sum() / (reg_mask.sum() + 1e-6)
        
        # Classification loss
        cls_loss = self.cls_loss(cls_pred, cls_target)
        cls_loss = cls_loss.mean()
        
        return reg_loss + cls_loss