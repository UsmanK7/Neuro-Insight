# neuro_insight_app/pdf_generator.py
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import io
import os
from django.conf import settings
from PIL import Image as PILImage
import base64
import tempfile

def generate_patient_report(patient_name, original_img_data, prediction_img_data, tumor_grade):
    """Generate a PDF report for the patient's MRI scan analysis"""
    
    # Create PDF buffer
    buffer = io.BytesIO()
    
    # Set up the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                            rightMargin=72, leftMargin=72, 
                            topMargin=72, bottomMargin=18)
    
    # Create a list to hold the PDF elements
    elements = []
    
    # Get the default stylesheet
    styles = getSampleStyleSheet()
    
    # Modify existing styles instead of adding new ones with duplicate names
    styles['Title'].fontName = 'Helvetica-Bold'
    styles['Title'].fontSize = 16
    styles['Title'].alignment = TA_CENTER
    styles['Title'].spaceAfter = 12
    
    # Modify Heading2 style instead of creating a new one
    styles['Heading2'].fontName = 'Helvetica-Bold'
    styles['Heading2'].fontSize = 14
    styles['Heading2'].spaceBefore = 12
    styles['Heading2'].spaceAfter = 6
    
    # Modify Normal style
    styles['Normal'].fontName = 'Helvetica'
    styles['Normal'].fontSize = 12
    styles['Normal'].spaceBefore = 6
    styles['Normal'].spaceAfter = 6
    
    # Add title
    elements.append(Paragraph("NeuroInsight Brain Tumor Analysis Report", styles['Title']))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add patient information
    elements.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles['Normal']))
    elements.append(Paragraph(f"<b>Tumor Grade:</b> {tumor_grade}", styles['Normal']))
    elements.append(Paragraph(f"<b>Report Date:</b> {__import__('datetime').datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    elements.append(Spacer(1, 0.25*inch))
    
    # Add MRI images
    elements.append(Paragraph("MRI Scan Images", styles['Heading2']))
    
    # Process and save the images temporarily
    temp_dir = tempfile.mkdtemp()
    
    # Extract base64 data and save original image
    original_img_binary = base64.b64decode(original_img_data)
    original_img_path = os.path.join(temp_dir, 'original.png')
    with open(original_img_path, 'wb') as f:
        f.write(original_img_binary)
    
    # Extract base64 data and save prediction image
    prediction_img_binary = base64.b64decode(prediction_img_data)
    prediction_img_path = os.path.join(temp_dir, 'prediction.png')
    with open(prediction_img_path, 'wb') as f:
        f.write(prediction_img_binary)
    
    # Create a table for the images
    img_width = 2.5 * inch  # Adjust as needed for your images
    
    # Load images with ReportLab
    original_img = Image(original_img_path, width=img_width, height=img_width)
    prediction_img = Image(prediction_img_path, width=img_width, height=img_width)
    
    # Create image captions
    original_caption = Paragraph("Original MRI Scan", styles['Normal'])
    prediction_caption = Paragraph("Segmented Tumor Prediction", styles['Normal'])
    
    # Create a table for the images and captions
    data = [[original_img, prediction_img], 
            [original_caption, prediction_caption]]
    
    t = Table(data, colWidths=[3*inch, 3*inch])
    t.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.white),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.white),
    ]))
    
    elements.append(t)
    elements.append(Spacer(1, 0.25*inch))
    
    # Add analysis details based on tumor grade
    elements.append(Paragraph("Analysis Summary", styles['Heading2']))
    
    if tumor_grade == "No Tumor":
        elements.append(Paragraph(
            "<b>Findings:</b> No abnormal tissue growth or tumor-like characteristics detected in the scanned area.", 
            styles['Normal']))
        elements.append(Paragraph(
            "<b>Recommendation:</b> No immediate treatment required. Routine follow-up may be recommended based on patient history.", 
            styles['Normal']))
    elif tumor_grade == "HGG":
        elements.append(Paragraph(
            "<b>Growth Rate:</b> Fast-growing, aggressive tumor that rapidly invades nearby brain tissue.", 
            styles['Normal']))
        elements.append(Paragraph(
            "<b>Typical Symptoms:</b> May include headaches, seizures, personality changes, and neurological deficits that worsen quickly.", 
            styles['Normal']))
        elements.append(Paragraph(
            "<b>Clinical Significance:</b> Requires urgent treatment. These tumors tend to be more resistant to treatment and have a higher recurrence rate.", 
            styles['Normal']))
    elif tumor_grade == "LGG":
        elements.append(Paragraph(
            "<b>Growth Rate:</b> Slow-growing tumor that tends to be less aggressive than high-grade gliomas.", 
            styles['Normal']))
        elements.append(Paragraph(
            "<b>Typical Symptoms:</b> May include seizures that are often the first symptom, with other neurological symptoms developing slowly over time.", 
            styles['Normal']))
        elements.append(Paragraph(
            "<b>Clinical Significance:</b> Generally has a better prognosis than high-grade gliomas. Treatment approach may be less aggressive depending on location and symptoms.", 
            styles['Normal']))
    
    # Add disclaimer
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("Disclaimer", styles['Heading2']))
    elements.append(Paragraph(
        "This report was generated by an AI-assisted tool", 
        styles['Normal']))
    
    # Build the PDF
    doc.build(elements)
    
    # Create thumbnail from prediction image for dashboard display
    pil_img = PILImage.open(prediction_img_path)
    pil_img = pil_img.resize((200, 200), PILImage.LANCZOS)  # Resize for thumbnail
    
    thumbnail_buffer = io.BytesIO()
    pil_img.save(thumbnail_buffer, format='PNG')
    thumbnail_data = thumbnail_buffer.getvalue()
    
    # Get the PDF data from the buffer
    pdf_data = buffer.getvalue()
    buffer.close()
    
    # Clean up temporary files
    os.remove(original_img_path)
    os.remove(prediction_img_path)
    os.rmdir(temp_dir)
    
    return pdf_data, thumbnail_data