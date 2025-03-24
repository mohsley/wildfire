import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import glob

def extract_trajectory_features(input_dir, output_dir, target_size=(224, 224)):
    """
    Extract and enhance trajectory features from HYSPLIT output images:
    1. Isolate the red trajectory line
    2. Create feature maps that highlight the trajectory path
    3. Standardize image size and format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all trajectory image files
    image_files = sorted(glob.glob(os.path.join(input_dir, "traj_hour_*.png")))
    
    if not image_files:
        print(f"No trajectory images found in {input_dir}")
        return
    
    # Initialize arrays to store features
    feature_sequence = np.zeros((len(image_files), target_size[0], target_size[1], 3), dtype=np.float32)
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing {os.path.basename(image_path)}...")
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read {image_path}, skipping...")
            continue
            
        # Convert to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target dimensions
        img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
        
        # Extract the red trajectory
        # Red channel is higher than blue and green channels in the trajectory
        red_mask = (img_resized[:,:,0] > 150) & (img_resized[:,:,0] > img_resized[:,:,1] + 50) & (img_resized[:,:,0] > img_resized[:,:,2] + 50)
        
        # Create a feature map highlighting the trajectory
        feature_map = np.zeros_like(img_resized)
        feature_map[red_mask] = [255, 0, 0]  # Mark trajectory as red
        
        # Create a distance transform to highlight proximity to the trajectory
        mask = red_mask.astype(np.uint8) * 255
        dist_transform = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        
        # Normalize distance transform to 0-1 range
        dist_norm = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
        
        # Create a 3-channel feature image:
        # Channel 1: Original image (grayscale) - geographical context
        # Channel 2: Trajectory mask - exact path
        # Channel 3: Distance transform - proximity feature
        
        # Convert original to grayscale and normalize
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY) / 255.0
        
        # Combine into feature channels
        feature_img = np.stack([
            gray,                           # Geographical context
            red_mask.astype(np.float32),    # Trajectory path
            1.0 - dist_norm                 # Proximity to trajectory (inverted)
        ], axis=-1)
        
        # Store in sequence
        feature_sequence[i] = feature_img
        
        # Save the feature image
        output_path = os.path.join(output_dir, f"feature_{os.path.basename(image_path)}")
        plt.imsave(output_path, feature_img)
        
        # Also save a visualization that's easier to interpret
        viz_img = np.zeros_like(img_resized)
        viz_img[:,:,0] = (1.0 - dist_norm) * 255  # Proximity as red channel
        viz_img[:,:,1] = red_mask.astype(np.uint8) * 255  # Path as green channel
        viz_img[:,:,2] = gray * 255  # Context as blue channel
        
        viz_path = os.path.join(output_dir, f"viz_{os.path.basename(image_path)}")
        cv2.imwrite(viz_path, cv2.cvtColor(viz_img, cv2.COLOR_RGB2BGR))
    
    # Save the feature sequence as numpy array
    np.save(os.path.join(output_dir, "trajectory_features.npy"), feature_sequence)
    print(f"Saved feature sequence with shape {feature_sequence.shape}")
    
    # Reshape for ConvLSTM input [samples, time_steps, height, width, channels]
    convlstm_input = np.expand_dims(feature_sequence, axis=0)
    np.save(os.path.join(output_dir, "convlstm_input.npy"), convlstm_input)
    print(f"Saved ConvLSTM input with shape {convlstm_input.shape}")
    
    return convlstm_input

def visualize_sequence(feature_sequence, output_dir):
    """Create a visualization of the feature sequence"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(feature_sequence.shape[0]):
        features = feature_sequence[i]
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original-like visualization
        combined = np.stack([
            features[:,:,1] * 255,  # Trajectory as red
            features[:,:,0] * 255,  # Context as green
            features[:,:,0] * 255   # Context as blue
        ], axis=-1).astype(np.uint8)
        
        axes[0].imshow(features[:,:,0], cmap='gray')
        axes[0].set_title("Geographic Context")
        axes[0].axis('off')
        
        axes[1].imshow(features[:,:,1], cmap='hot')
        axes[1].set_title("Trajectory Path")
        axes[1].axis('off')
        
        axes[2].imshow(features[:,:,2], cmap='viridis')
        axes[2].set_title("Proximity Feature")
        axes[2].axis('off')
        
        axes[3].imshow(combined)
        axes[3].set_title("Combined Visualization")
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sequence_viz_{i+1:02d}.png"))
        plt.close()

if __name__ == "__main__":
    # Define directories
    input_dir = "output"  # Directory with your trajectory images
    output_dir = "convlstm_features"  # Directory to save processed features
    viz_dir = "visualizations"  # Directory for visualizations
    
    # Process the images
    feature_sequence = extract_trajectory_features(input_dir, output_dir)
    
    # Create visualizations
    if feature_sequence is not None:
        visualize_sequence(feature_sequence[0], viz_dir)  # Unpack the first dimension for visualization