# tumor_segmentation/utils.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import nibabel as nib
from io import BytesIO
import base64
from matplotlib.colors import to_rgb
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import MeanIoU

# Custom metrics for the segmentation model
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

# Global class information for coloring
CLASS_INFO = {
    0: {'name': 'No Tumor', 'color': 'black'},
    1: {'name': 'Necrotic Core', 'color': '#FF4500'}, 
    2: {'name': 'Edema', 'color': '#18B880'},
    4: {'name': 'Enhancing', 'color': '#E6D74F'}  # original class 4, mapped from 3
}

class BrainTumorPredictor:
    def __init__(self, seg_model_path, class_model_path):
        # Load segmentation model with custom metrics
        self.seg_model = tf.keras.models.load_model(
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
        self.class_model = tf.keras.models.load_model(
            class_model_path,
            custom_objects={
                'sensitivity': sensitivity,
                'specificity': specificity
            }
        )
    
    def preprocess_modalities(self, t1ce_slice, flair_slice):
        """Preprocess MRI slices for segmentation model"""
        t1ce = (t1ce_slice - np.mean(t1ce_slice)) / (np.std(t1ce_slice) + 1e-8)
        flair = (flair_slice - np.mean(flair_slice)) / (np.std(flair_slice) + 1e-8)

        t1ce_resized = cv2.resize(t1ce, (128, 128), interpolation=cv2.INTER_LINEAR)
        flair_resized = cv2.resize(flair, (128, 128), interpolation=cv2.INTER_LINEAR)

        input_data = np.stack([t1ce_resized, flair_resized], axis=-1)
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        return input_data
    
    def predict_segmentation(self, t1ce_slice, flair_slice):
        """Predict segmentation mask"""
        input_data = self.preprocess_modalities(t1ce_slice, flair_slice)
        prediction = self.seg_model.predict(input_data)
        prediction_mask = np.argmax(prediction[0], axis=-1)
        return prediction_mask
    
    def colorize_prediction(self, pred_mask):
        """Convert prediction mask to RGB image"""
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
    
    def predict_grade(self, t1ce_slice, flair_slice, img_size=(240, 240)):
        """Predict tumor grade (HGG or LGG)"""
        # Resize
        t1ce = cv2.resize(t1ce_slice, img_size)
        flair = cv2.resize(flair_slice, img_size)

        # Normalize
        t1ce = t1ce / np.max(t1ce) if np.max(t1ce) != 0 else t1ce
        flair = flair / np.max(flair) if np.max(flair) != 0 else flair

        # Create 3-channel image: repeat one modality or stack modalities
        img = np.stack([t1ce, flair, t1ce], axis=-1)  # Shape: (H, W, 3)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = self.class_model.predict(img)[0][0]

        # Convert to class
        return "HGG" if pred >= 0.5 else "LGG"
    
    def segment_and_classify(self, t1ce_path, flair_path, slice_idx):
        """Segment tumor and classify its grade"""
        # Load MRI data
        t1ce_img = nib.load(t1ce_path).get_fdata()
        flair_img = nib.load(flair_path).get_fdata()

        t1ce_slice = t1ce_img[:, :, slice_idx]
        flair_slice = flair_img[:, :, slice_idx]

        # Preview images for labels div
        preview = self.generate_preview_plot(t1ce_slice, flair_slice)
        
        # Predict segmentation mask
        pred_mask = self.predict_segmentation(t1ce_slice, flair_slice)
        seg_image = self.colorize_prediction(pred_mask)

        # Check if tumor is detected
        if np.all(pred_mask == 0):  # All pixels are background
            grade = "No Tumor"
        else:
            # Predict tumor grade if present
            grade = self.predict_grade(t1ce_slice, flair_slice)

        # Generate prediction visualization
        prediction_img = self.generate_prediction_plot(t1ce_slice, flair_slice, seg_image, grade)
        
        return preview, prediction_img, grade
    
    def generate_preview_plot(self, t1ce_slice, flair_slice):
        """Generate preview plot of the MRI slices"""
        plt.figure(figsize=((5, 10)))

        plt.subplot(2, 1, 1)
        plt.imshow(t1ce_slice, cmap='gray')
        plt.title("T1CE")
        plt.axis('off')

        plt.subplot(2, 1, 2)
        plt.imshow(flair_slice, cmap='gray')
        plt.title("FLAIR")
        plt.axis('off')

        plt.tight_layout()
        
        # Save plot to a base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        graphic = base64.b64encode(image_png).decode('utf-8')
        return graphic
    
    def generate_prediction_plot(self, t1ce_slice, flair_slice, seg_image, grade):
        """Generate plot with only the segmentation result and more visible legend"""
        plt.figure(figsize=(10, 8))  # Larger figure to make room for the clear legend

        # Show only the Predicted Segmentation Mask 
        plt.imshow(seg_image)
        plt.title(f"Predicted Segmentation - Grade: {grade}", fontsize=14)
        plt.axis('off')

        # Add color legend with improved visibility
        labels = list(CLASS_INFO.values())
        handles = []
        for label in labels:
            color = np.array(to_rgb(label['color'])) * 255
            handle = plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color/255, markersize=20, 
                                label=label['name'])
            handles.append(handle)

        # Customize the legend for better visibility
        plt.legend(
            handles=handles,
            loc='upper right',
            bbox_to_anchor=(1.25, 0.99),
            title="Tumor Labels",
            title_fontsize=16,
            fontsize=14,
            borderpad=1.5,
            frameon=True,
            facecolor='white',
            edgecolor='black',
            borderaxespad=1.5,
            shadow=True
        )

        plt.tight_layout()
    
        # Save plot to a base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
    
        graphic = base64.b64encode(image_png).decode('utf-8')
        return graphic

