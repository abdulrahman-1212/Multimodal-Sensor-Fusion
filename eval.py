from transFuser import TransFuser, Config, KITTICollator
from preprocessing import KITTIDataset, KITTITransform
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.functional as F
import os

config = Config()

def evaluate_model(model_path, output_dir='results'):
    # Load model
    model = TransFuser(num_classes=config.num_classes).to(config.device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Prepare validation dataset
    all_samples = [f.split('.')[0] for f in os.listdir(os.path.join(config.data_root, 'image_2')) 
                  if f.endswith('.png')]
    _, val_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
    
    val_dataset = KITTIDataset(
        root_dir=config.data_root,
        samples=val_samples,
        transform=KITTITransform(augment=False),
        is_train=False
    )
    
    collator = KITTICollator()
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluation loop
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(config.device)
            point_clouds = batch['point_cloud'].to(config.device)
            sample_names = batch['sample_name']
            
            # Get predictions
            reg_preds, cls_preds = model(images, point_clouds)
            
            # Convert to KITTI format
            for i, sample_name in enumerate(sample_names):
                # Get class predictions
                class_probs = F.softmax(cls_preds[i], dim=-1)
                class_ids = torch.argmax(class_probs, dim=-1)
                
                # Get regression predictions
                pred_boxes = reg_preds[i].cpu().numpy()
                
                # Write to file
                with open(os.path.join(output_dir, f'{sample_name}.txt'), 'w') as f:
                    for box_idx in range(pred_boxes.shape[0]):
                        class_id = class_ids[box_idx].item()
                        if class_id == 0:  # Skip background
                            continue
                            
                        class_name = ['Car', 'Pedestrian', 'Cyclist'][class_id - 1]
                        box = pred_boxes[box_idx]
                        
                        # Format: type, trunc, occ, alpha, bbox, dimensions, location, rotation_y, score
                        f.write(
                            f"{class_name} -1 -1 -10 "  # Placeholder values for unused fields
                            f"-1 -1 -1 -1 "  # Placeholder bbox
                            f"{box[3]:.2f} {box[4]:.2f} {box[5]:.2f} "  # dimensions (h,w,l)
                            f"{box[0]:.2f} {box[1]:.2f} {box[2]:.2f} "  # location (x,y,z)
                            f"{box[6]:.2f} "  # rotation_y
                            f"{class_probs[box_idx, class_id]:.2f}\n"  # confidence
                        )

# To run evaluation:
evaluate_model('best_transfuser_kitti.pth', 'kitti_results')