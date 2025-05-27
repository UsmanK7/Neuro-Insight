# tumor_app/ml_model/model_utils.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import backend as K
import io
import base64
from matplotlib.colors import to_rgb
import os
from django.conf import settings

# -------------------- Custom Metrics --------------------
def dice_coef(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred))
    return (2. * intersection) / (K.sum(K.square(y_true)) + K.sum(K.square(y_pred)) + epsilon)

def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    return dice_coef(y_true[:,:,:,1], y_pred[:,:,:,1], epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    return dice_coef(y_true[:,:,:,2], y_pred[:,:,:,2], epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    return dice_coef(y_true[:,:,:,3], y_pred[:,:,:,3], epsilon)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def iou(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return K.mean((intersection + K.epsilon()) / (union + K.epsilon()))

# -------------------- CLASS INFO -------------------- 
CLASS_INFO = {
    0: {'name': 'No Tumor', 'color': 'black'},
    1: {'name': 'Necrotic Core', 'color': '#440054'}, 
    2: {'name': 'Edema', 'color': '#18B880'},
    4: {'name': 'Enhancing', 'color': '#E6D74F'}  # original class 4, mapped from 3
}

# -------------------- Load Models --------------------
def load_models():
    # Get paths relative to Django project
    seg_model_path = os.path.join(settings.BASE_DIR, 'tumor_app/ml_model/seg_model.keras')
    clf_model_path = os.path.join(settings.BASE_DIR, 'tumor_app/ml_model/best_model.keras')
    
    # Load segmentation model
    seg_model = load_model(
        seg_model_path, 
        custom_objects={
            "dice_coef": dice_coef,
            "dice_coef_necrotic": dice_coef_necrotic,
            "dice_coef_edema": dice_coef_edema,
            "dice_coef_enhancing": dice_coef_enhancing,
            "precision": precision,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "iou": iou,
            "MeanIoU": MeanIoU
        }
    )
    
    # Load classification model
    clf_model = load_model(
        clf_model_path, 
        custom_objects={
            'sensitivity': sensitivity, 
            'specificity': specificity
        }
    )
    
    return seg_model, clf_model

# -------------------- Preprocess Modalities --------------------
def preprocess_modalities(t1ce_slice, flair_slice):
    t1ce = (t1ce_slice - np.mean(t1ce_slice)) / (np.std(t1ce_slice) + 1e-8)
    flair = (flair_slice - np.mean(flair_slice)) / (np.std(flair_slice) + 1e-8)

    t1ce_resized = cv2.resize(t1ce, (128, 128), interpolation=cv2.INTER_LINEAR)
    flair_resized = cv2.resize(flair, (128, 128), interpolation=cv2.INTER_LINEAR)

    input_data = np.stack([t1ce_resized, flair_resized], axis=-1)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    return input_data

# -------------------- Predict Segmentation --------------------
def predict_segmentation(model, t1ce_slice, flair_slice):
    input_data = preprocess_modalities(t1ce_slice, flair_slice)
    prediction = model.predict(input_data)
    prediction_mask = np.argmax(prediction[0], axis=-1)
    return prediction_mask

# -------------------- Colorize Prediction --------------------
def colorize_prediction(pred_mask):
    height, width = pred_mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Remap class 3 → 4 for visualization
    pred_mask = pred_mask.copy()
    pred_mask[pred_mask == 3] = 4

    for class_id, info in CLASS_INFO.items():
        rgb = np.array(to_rgb(info['color'])) * 255  # convert to 0–255 range
        color = tuple(rgb.astype(np.uint8))
        color_mask[pred_mask == class_id] = color

    return color_mask

# -------------------- Predict Grade --------------------
def predict_grade(model, t1ce_slice, flair_slice, img_size=(240, 240)):
    """
    Predicts HGG vs LGG using trained model.
    """
    # Resize
    t1ce = cv2.resize(t1ce_slice, img_size)
    flair = cv2.resize(flair_slice, img_size)

    # Normalize
    t1ce = t1ce / np.max(t1ce) if np.max(t1ce) != 0 else t1ce
    flair = flair / np.max(flair) if np.max(flair) != 0 else flair

    # Create 3-channel image
    img = np.stack([t1ce, flair, t1ce], axis=-1)  # Shape: (H, W, 3)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)[0][0]

    # Convert to class
    return "HGG" if pred >= 0.5 else "LGG"

# -------------------- Generate Plot --------------------
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

# -------------------- Main Processing Function --------------------
def process_mri(t1ce_path, flair_path, slice_idx):
    """
    Process MRI scans and return the results.
    
    Args:
        t1ce_path (str): Path to T1CE .nii file
        flair_path (str): Path to FLAIR .nii file
        slice_idx (int): Slice number to process

    Returns:
        dict: Dictionary containing results
    """
    # Load models
    seg_model, clf_model = load_models()
    
    # Load MRI data
    t1ce_img = nib.load(t1ce_path).get_fdata()
    flair_img = nib.load(flair_path).get_fdata()

    # Get specified slice
    t1ce_slice = t1ce_img[:, :, slice_idx]
    flair_slice = flair_img[:, :, slice_idx]

    # Predict segmentation mask
    pred_mask = predict_segmentation(seg_model, t1ce_slice, flair_slice)
    seg_image = colorize_prediction(pred_mask)

    # Check if tumor is detected
    if np.all(pred_mask == 0):  # All pixels are background
        tumor_detected = False
        grade = "No Tumor"
    else:
        tumor_detected = True
        # Predict tumor grade
        grade = predict_grade(clf_model, t1ce_slice, flair_slice)

    # Generate visualization
    plot_img = generate_plots(t1ce_slice, flair_slice, seg_image, grade)
    
    # Generate individual images for the preview section
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