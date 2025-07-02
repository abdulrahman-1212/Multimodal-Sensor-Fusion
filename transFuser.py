import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import ViTModel
import open3d as o3d
from sklearn.model_selection import train_test_split

from preprocessing import KITTIDataset, KITTITransform, KITTICollator, KITTILoss

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class Config:
    def __init__(self):
        self.data_root = 'kitti_dataset'  # Update with your KITTI path
        self.batch_size = 8
        self.num_workers = 4
        self.learning_rate = 1e-4
        self.num_epochs = 50
        self.num_classes = 3  # Car, Pedestrian, Cyclist
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_points = 40000  # Max points in point cloud after sampling
        self.image_size = (224, 224)  # ViT input size

config = Config()



## 3. Complete Model Architecture
class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.conv1x1 = nn.Conv2d(768, 256, 1)
        self.norm = nn.BatchNorm2d(256)
        
    def forward(self, x):
        # Forward pass through ViT
        outputs = self.vit(x)
        features = outputs.last_hidden_state
        
        # Reshape from sequence to spatial
        batch_size = features.size(0)
        cls_token = features[:, 0]  # [B, 768]
        patch_tokens = features[:, 1:]  # [B, 196, 768]
        
        # Reshape to 2D feature map (14x14)
        spatial_features = patch_tokens.permute(0, 2, 1).view(batch_size, 768, 14, 14)
        
        # Project to 256 channels
        spatial_features = F.relu(self.norm(self.conv1x1(spatial_features)))
        
        return spatial_features, cls_token

class PointCloudBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x: [B, N, 3]
        batch_size, num_points, _ = x.size()
        
        # Process each point
        x = x.view(-1, 3)  # [B*N, 3]
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        
        # Reshape back to [B, N, 256]
        x = x.view(batch_size, num_points, -1)
        
        # Global feature
        global_feature = torch.max(x, dim=1)[0]  # [B, 256]
        
        return x, global_feature

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, query, key, value):
        # query: [B, N, D]
        # key, value: [B, M, D]
        batch_size = query.size(0)
        
        # Project to query, key, value
        q = self.q_proj(query)  # [B, N, D]
        k = self.k_proj(key)     # [B, M, D]
        v = self.v_proj(value)   # [B, M, D]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output

class TransFuser(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.image_backbone = ImageBackbone()
        self.pc_backbone = PointCloudBackbone()
        
        # Cross-modal attention
        self.img2pc_attn = CrossModalAttention(256)
        self.pc2img_attn = CrossModalAttention(256)
        
        # Feature projection
        self.img_proj = nn.Linear(256, 256)
        self.pc_proj = nn.Linear(256, 256)
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes * 7)  # 7 parameters per detection
        )
        
        # Classification head
        self.class_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, image, point_cloud):
        # Extract features
        img_features, img_global = self.image_backbone(image)  # [B, 256, 14, 14], [B, 768]
        pc_features, pc_global = self.pc_backbone(point_cloud)  # [B, N, 256], [B, 256]
        
        # Reshape image features for attention
        B, C, H, W = img_features.shape
        img_features = img_features.view(B, C, -1).permute(0, 2, 1)  # [B, HW, 256]
        
        # Cross-modal attention
        fused_img = self.pc2img_attn(img_features, pc_features, pc_features)  # [B, HW, 256]
        fused_pc = self.img2pc_attn(pc_features, img_features, img_features)  # [B, N, 256]
        
        # Global average pooling
        img_global = fused_img.mean(dim=1)  # [B, 256]
        pc_global = fused_pc.mean(dim=1)    # [B, 256]
        
        # Project features
        img_global = F.relu(self.img_proj(img_global))
        pc_global = F.relu(self.pc_proj(pc_global))
        
        # Concatenate features
        combined = torch.cat([img_global, pc_global], dim=1)  # [B, 512]
        
        # Predictions
        detections = self.detection_head(combined).view(B, -1, 7)
        class_logits = self.class_head(combined)
        
        return detections, class_logits



## 5. Training Loop
def train_model():
    # Prepare datasets
    all_samples = [f.split('.')[0] for f in os.listdir(os.path.join(config.data_root, 'image_2')) 
                  if f.endswith('.png')]
    train_samples, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
    
    train_dataset = KITTIDataset(
        root_dir=config.data_root,
        samples=train_samples,
        transform=KITTITransform(augment=True),
        is_train=True
    )
    
    val_dataset = KITTIDataset(
        root_dir=config.data_root,
        samples=val_samples,
        transform=KITTITransform(augment=False),
        is_train=False
    )
    
    # Create data loaders
    collator = KITTICollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator
    )
    
    # Initialize model
    model = TransFuser(num_classes=config.num_classes).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = KITTILoss().to(config.device)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(config.device)
            point_clouds = batch['point_cloud'].to(config.device)
            reg_targets = batch['target'].to(config.device)
            cls_targets = batch['class_label'].to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            reg_preds, cls_preds = model(images, point_clouds)
            
            # Compute loss
            loss = criterion((reg_preds, cls_preds), (reg_targets, cls_targets))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(config.device)
                point_clouds = batch['point_cloud'].to(config.device)
                reg_targets = batch['target'].to(config.device)
                cls_targets = batch['class_label'].to(config.device)
                
                reg_preds, cls_preds = model(images, point_clouds)
                loss = criterion((reg_preds, cls_preds), (reg_targets, cls_targets))
                val_loss += loss.item()
        
        # Print statistics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{config.num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transfuser_kitti.pth')
            print('Saved new best model')

## 6. Main Execution
if __name__ == '__main__':
    train_model()
    
