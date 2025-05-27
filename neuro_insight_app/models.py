# neuro_insight_app/models.py
from django.db import models
import os
from django.contrib.auth.models import User

class MRIScan(models.Model):
    t1ce = models.FileField(upload_to='mri_scans/t1ce/')
    flair = models.FileField(upload_to='mri_scans/flair/')
    slice_number = models.IntegerField(default=70)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"MRI Scan {self.id} - Slice {self.slice_number}"
    
    def delete(self, *args, **kwargs):
        # Delete the files when the model instance is deleted
        if self.t1ce:
            if os.path.isfile(self.t1ce.path):
                os.remove(self.t1ce.path)
        if self.flair:
            if os.path.isfile(self.flair.path):
                os.remove(self.flair.path)
        super().delete(*args, **kwargs)

class PatientReport(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='patient_reports')
    patient_name = models.CharField(max_length=100)
    pdf_report = models.FileField(upload_to='patient_reports/')
    thumbnail = models.ImageField(upload_to='report_thumbnails/', blank=True, null=True)
    tumor_grade = models.CharField(max_length=20, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.patient_name} - {self.tumor_grade}"
    
    def delete(self, *args, **kwargs):
        # Delete associated files when record is deleted
        if self.pdf_report:
            if os.path.isfile(self.pdf_report.path):
                os.remove(self.pdf_report.path)
        if self.thumbnail:
            if os.path.isfile(self.thumbnail.path):
                os.remove(self.thumbnail.path)
        super().delete(*args, **kwargs)