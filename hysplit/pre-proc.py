import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import cv2

# Define parameters
INPUT_DIR = "output"  # Directory with your trajectory images
OUTPUT_DIR = "preprocessed"  # Directory to save processed images
TARGET_SIZE = (224, 224)  # Common input size for CNNs (can be adjusted)
SEQUENCE_LENGTH = 24  # Number of frames (hours) in your sequence

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to preprocess a single image
def preprocess_image(image_path, target_size=TARGET_SIZE):
    """
    Preprocess image for ConvLSTM:
    - Resize to target dimensions
    - Convert to RGB if needed
    - Normalize pixel values to [0,1]
    - Focus on the trajectory part
    """
    # Load the image
    img = Image.open(image_path)
    
    # Convert to RGB if in another mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to target dimensions
    img = img.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    return img_array

# Process trajectory images
def process_trajectory_images():
    # Get all trajectory image files and sort them
    image_files = sorted(glob.glob(os.path.join(INPUT_DIR, "traj_hour_*.png")))
    
    if not image_files:
        print("No trajectory images found in the input directory!")
        return None
    
    # Initialize a tensor to hold the sequence
    sequence = np.zeros((len(image_files), TARGET_SIZE[0], TARGET_SIZE[1], 3), dtype=np.float32)
    
    # Process each image
    for i, image_path in enumerate(image_files):
        processed_img = preprocess_image(image_path)
        sequence[i] = processed_img
        
        # Save the processed image
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
        plt.imsave(output_path, processed_img)
        
        print(f"Processed and saved: {output_path}")
    
    # Save the sequence as a NumPy array
    np.save(os.path.join(OUTPUT_DIR, "trajectory_sequence.npy"), sequence)
    print(f"Saved trajectory sequence with shape: {sequence.shape}")
    
    return sequence

# Optional: Create a function to enhance the trajectory
def enhance_trajectory(image_path, target_size=TARGET_SIZE):
    """
    Enhance the trajectory in the image:
    - Isolate the red trajectory
    - Improve contrast
    - Remove background noise
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to RGB (OpenCV uses BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a mask for red trajectory (adjust thresholds as needed)
    lower_red = np.array([180, 0, 0])
    upper_red = np.array([255, 80, 80])
    mask = cv2.inRange(img_rgb, lower_red, upper_red)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank image to draw the trajectory
    enhanced_img = np.zeros(img.shape, dtype=np.uint8)
    
    # Draw the contours on the blank image
    cv2.drawContours(enhanced_img, contours, -1, (255, 0, 0), 2)
    
    # Resize to target size
    enhanced_img = cv2.resize(enhanced_img, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return enhanced_img

# Function to prepare data for ConvLSTM
def prepare_convlstm_data(sequence, timesteps=SEQUENCE_LENGTH):
    """
    Format the sequence data for ConvLSTM input:
    - Reshape to [samples, timesteps, height, width, channels]
    """
    # If we have a single sequence
    if sequence.shape[0] == timesteps:
        # Reshape to [1, timesteps, height, width, channels]
        return np.expand_dims(sequence, axis=0)
    
    # If we have multiple sequences (not in this case, but for completeness)
    else:
        sequences = []
        for i in range(0, len(sequence) - timesteps + 1):
            sequences.append(sequence[i:i+timesteps])
        return np.array(sequences)

# Main execution
if __name__ == "__main__":
    # Process the trajectory images
    sequence = process_trajectory_images()
    
    if sequence is not None:
        # Prepare data for ConvLSTM
        convlstm_data = prepare_convlstm_data(sequence)
        print(f"ConvLSTM input shape: {convlstm_data.shape}")
        
        # Save the prepared data
        np.save(os.path.join(OUTPUT_DIR, "convlstm_input.npy"), convlstm_data)
        print(f"Saved ConvLSTM input data")
        
        # Create a sample ConvLSTM model structure (for reference)
        print("\nSample ConvLSTM model structure:")
        model = tf.keras.Sequential([
            tf.keras.layers.ConvLSTM2D(
                filters=64, kernel_size=(3, 3), padding='same',
                return_sequences=True, activation='relu',
                input_shape=(SEQUENCE_LENGTH, TARGET_SIZE[0], TARGET_SIZE[1], 3)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
            
            tf.keras.layers.ConvLSTM2D(
                filters=32, kernel_size=(3, 3), padding='same',
                return_sequences=True, activation='relu'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
        ])
        
        model.summary()