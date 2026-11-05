import os
import shutil
import random

source_dir = "./raw_data"

target_dir = "./cv_exercise_project" 

classes = ['pushup', 'squat', 'standing']

train_ratio = 0.70
val_ratio = 0.15

random.seed(42)

for cls in classes:
    src_folder = os.path.join(source_dir, cls) 
    
    if not os.path.exists(src_folder):
        print(f"Folder {src_folder} not found, skipping.")
        continue

    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    
    total = len(images)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train + n_val],
        'test': images[n_train + n_val:]
    }
    
    for split_name, split_files in splits.items():
        dst_folder = os.path.join(target_dir, split_name, cls)
        os.makedirs(dst_folder, exist_ok=True)
        for img in split_files:
            shutil.copy(os.path.join(src_folder, img), os.path.join(dst_folder, img))
            
    print(f"Class '{cls}' ({total} photo) copied to '{target_dir}' -> Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

