# ============================================================
# CELL 6 — VISUALIZE RE-ID CAPABILITIES (Nearest Neighbors)
# ============================================================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

def visualize_reid(model, dataset, num_queries=5, top_k=5):
    print("Extracting gallery embeddings for visualization (takes ~10 seconds)...")
    model.eval()
    
    # We will pick a random subset to be our "Gallery" to search through
    gallery_size = min(1000, len(dataset.samples))
    gallery_indices = random.sample(range(len(dataset.samples)), gallery_size)
    
    gallery_embeddings = []
    gallery_images = []
    
    # Process the gallery
    with torch.no_grad():
        for i in tqdm(gallery_indices, desc="Building Gallery"):
            sample = dataset.samples[i]
            img_path = os.path.join(dataset.img_dir, dataset.img_map[sample['image_id']])
            image = Image.open(img_path).convert("RGB")
            
            x, y, w, h = [int(v) for v in sample['bbox']]
            x, y = max(0, x-10), max(0, y-10)
            crop = image.crop((x, y, x+w+20, y+h+20))
            
            # Save original crop for plotting
            gallery_images.append(crop)
            
            # Extract Embedding
            tensor = dataset.augmentations(crop).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embedding = model(tensor).to(torch.float32).cpu().numpy()
            gallery_embeddings.append(embedding[0])
            
    gallery_embeddings = np.array(gallery_embeddings)
    
    # Pick random queries that are NOT in the gallery
    query_indices = [i for i in range(len(dataset.samples)) if i not in gallery_indices]
    queries = random.sample(query_indices, num_queries)
    
    # Setup Plot
    fig, axes = plt.subplots(num_queries, top_k + 1, figsize=(15, 3 * num_queries))
    fig.suptitle("DINOv3 Re-ID Matches: Query Image vs Top-5 Matches in Gallery", fontsize=16)
    
    with torch.no_grad():
        for row, q_idx in enumerate(queries):
            # 1. Load Query
            sample = dataset.samples[q_idx]
            img_path = os.path.join(dataset.img_dir, dataset.img_map[sample['image_id']])
            image = Image.open(img_path).convert("RGB")
            x, y, w, h = [int(v) for v in sample['bbox']]
            x, y = max(0, x-10), max(0, y-10)
            query_crop = image.crop((x, y, x+w+20, y+h+20))
            
            # Extract Embedding
            q_tensor = dataset.augmentations(query_crop).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                q_emb = model(q_tensor).to(torch.float32).cpu().numpy()
            
            # 2. Calculate Cosine Similarity against entire gallery
            similarities = cosine_similarity(q_emb, gallery_embeddings)[0]
            
            # 3. Get Top-K highest scores
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # 4. Plot Query
            ax = axes[row, 0]
            ax.imshow(query_crop)
            ax.set_title("QUERY", color='blue', fontweight='bold')
            ax.axis('off')
            
            # 5. Plot Top-K Matches
            for col, match_idx in enumerate(top_indices):
                score = similarities[match_idx]
                ax = axes[row, col + 1]
                ax.imshow(gallery_images[match_idx])
                
                # Green text if highly confident (>0.5), Orange otherwise
                color = 'green' if score > 0.50 else 'orange'
                ax.set_title(f"Sim: {score:.2f}", color=color)
                ax.axis('off')
                
    plt.tight_layout()
    plt.show()

# Run the visualization!
visualize_reid(model, dataset)
