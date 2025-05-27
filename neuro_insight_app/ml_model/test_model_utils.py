# tumor_app/ml_model/test_model_utils.py
"""
A simplified version of model_utils.py that doesn't require the actual models.
Use this during development to test the UI functionality.
"""

import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.colors import to_rgb
import os
import time
import random

# Mock CLASS_INFO for visualization
CLASS_INFO = {
    0: {'name': 'No Tumor', 'color': 'black'},
    1: {'name': 'Necrotic Core', 'color': '#440054'}, 
    2: {'name': 'Edema', 'color': '#18B880'},
    4: {'name': 'Enhancing', 'color': '#E6D74F'}
}

def process_mri(t1ce_path, flair_path, slice_idx):
    """
    Mock function to process MRI scans for testing the UI.
    
    Args:
        t1ce_path (str): Path to T1CE .nii file
        flair_path (str): Path to FLAIR .nii file
        slice_idx (int): Slice number to process

    Returns:
        dict: Dictionary containing results
    """
    # Simulate processing delay
    time.sleep(3)
    
    # Check if files exist (at least check this)
    if not os.path.exists(t1ce_path) or not os.path.exists(flair_path):
        raise FileNotFoundError(f"One or both MRI files not found: {t1ce_path}, {flair_path}")
    
    # Create mock slices with random shapes and values
    t1ce_slice = np.random.rand(240, 240)
    flair_slice = np.random.rand(240, 240)
    
    # Create a simple mock segmentation mask
    # 80% chance of "tumor" for testing
    if random.random() < 0.8:
        tumor_detected = True
        # Create a circular "tumor" in the center
        height, width = t1ce_slice.shape
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height / 2, width / 2
        radius = min(height, width) / 4
        
        # Create mask for different tumor regions
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Create segmentation mask with tumor classes
        seg_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Inner core (necrotic) - class 1
        inner_radius = radius * 0.5
        inner_mask = (x - center_x)**2 + (y - center_y)**2 <= inner_radius**2
        seg_mask[inner_mask] = 1
        
        # Edema - class 2
        edema_radius = radius * 1.5
        edema_mask = (x - center_x)**2 + (y - center_y)**2 <= edema_radius**2
        seg_mask[edema_mask & ~mask] = 2
        
        # Enhancing tumor - class 3 (will be mapped to 4)
        enhancing_mask = mask & ~inner_mask
        seg_mask[enhancing_mask] = 3
        
        # Randomly choose between HGG and LGG
        grade = "HGG" if random.random() < 0.5 else "LGG"
    else:
        # No tumor
        tumor_detected = False
        seg_mask = np.zeros(t1ce_slice.shape, dtype=np.uint8)
        grade = "No Tumor"
    
    # Create colored segmentation image
    seg_image = colorize_prediction(seg_mask)
    
    # Generate visualizations
    plot_img = generate_plots(t1ce_slice, flair_slice, seg_image, grade)
    
    # Generate individual preview images
    plt.figure(figsize=(6, 6))
    plt.imshow(t1ce_slice, cmap='gray')
    plt.title("T1CE")
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    t1ce_preview = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(flair_slice, cmap='gray')
    plt.title("FLAIR")
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    flair_preview = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return {
        'tumor_detected': tumor_detected,
        'grade': grade,
        'plot_img': plot_img,
        't1ce_preview': t1ce_preview,
        'flair_preview': flair_preview
    }

def colorize_prediction(pred_mask):
    """Convert prediction mask to RGB image"""
    height, width = pred_mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Remap class 3 → 4 for visualization
    pred_mask = pred_mask.copy()
    pred_mask[pred_mask == 3] = 4

    for class_id, info in CLASS_INFO.items():
        rgb = np.array(to_rgb(info['color'])) * 255  # convert to 0–255 range
        color = rgb.astype(np.uint8)
        mask = pred_mask == class_id
        color_mask[mask] = color

    return color_mask

def generate_plots(t1ce_slice, flair_slice, seg_image, grade):
    """Generate plots and return as base64 encoded images"""
    plt.figure(figsize=(18, 6))

    # Show T1CE Slice
    plt.subplot(1, 3, 1)
    plt.imshow(t1ce_slice, cmap='gray')
    plt.title("T1CE")
    plt.axis('off')

    # Show FLAIR Slice
    plt.subplot(1, 3, 2)
    plt.imshow(flair_slice, cmap='gray')
    plt.title("FLAIR")
    plt.axis('off')

    # Show Predicted Segmentation Mask
    plt.subplot(1, 3, 3)
    plt.imshow(seg_image)
    plt.title(f"Predicted Segmentation - Grade: {grade}")
    plt.axis('off')

    # Add color legend
    labels = list(CLASS_INFO.values())
    handles = []
    for label in labels:
        color = np.array(to_rgb(label['color'])) * 255
        handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color/255, markersize=10, label=label['name'])
        handles.append(handle)

    # Customize the legend
    plt.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=(1.2, 1),
        title="Tumor Labels",
        fontsize=12,
        borderpad=1.0,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        borderaxespad=1.0
    )

    plt.tight_layout()
    
    # Save plot to in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    # Convert buffer to base64 string
    plot_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return plot_img