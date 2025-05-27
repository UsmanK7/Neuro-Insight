# neuro_insight_app/views.py
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from .models import MRIScan, PatientReport
from .forms import SignUpForm, LoginForm, PatientNameForm
from .utils import BrainTumorPredictor
from .pdf_generator import generate_patient_report
import json
from django.db import IntegrityError
from django.core.files.base import ContentFile
import base64
def home_screen(request):
    """Our team page view - accessible without login"""
    
    return render(request, 'neuro_insight_app/home.html')
def our_team(request):
    """Our team page view - accessible without login"""
    return render(request, 'neuro_insight_app/our_team.html')



def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            try:
                user = form.save()
                login(request, user)
                return redirect('neuro_insight_app:home')
            except IntegrityError:
                # Handle the unique constraint error
                form.add_error('username', 'This username is already taken. Please choose a different one.')
    else:
        form = SignUpForm()
    return render(request, 'neuro_insight_app/signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = LoginForm(data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                # Redirect to a success page
                return redirect('neuro_insight_app:home')
    else:
        form = LoginForm()
    return render(request, 'neuro_insight_app/login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('neuro_insight_app:login')

@login_required(login_url='neuro_insight_app:login')
def home(request):
    """Home page view"""
    patient_form = PatientNameForm()
    return render(request, 'neuro_insight_app/index.html', {'patient_form': patient_form})

@login_required(login_url='neuro_insight_app:login')
def upload_mri(request):
    """Handle MRI file uploads"""
    if request.method == 'POST':
        # Delete previous scan if it exists
        MRIScan.objects.all().delete()

        # Create new scan record
        scan = MRIScan(
            t1ce=request.FILES['t1ce'],
            flair=request.FILES['flair'],
            slice_number=request.POST.get('slice_number', 70)
        )
        scan.save()

        # Return success response
        return JsonResponse({
            'status': 'success',
            'scan_id': scan.id,
            't1ce_path': scan.t1ce.url,
            'flair_path': scan.flair.url,
            'slice_number': scan.slice_number
        })

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@login_required(login_url='neuro_insight_app:login')
def predict(request):
    """Process MRI scans and return prediction results"""
    if request.method == 'POST':
        try:
            # Get the latest scan
            scan = MRIScan.objects.latest('uploaded_at')

            # Initialize predictor
            seg_model_path = os.path.join(settings.BASE_DIR, 'models', 'seg_model.keras')
            class_model_path = os.path.join(settings.BASE_DIR, 'models', 'class_model.keras')
            predictor = BrainTumorPredictor(seg_model_path, class_model_path)

            # Perform prediction
            preview_img, prediction_img, grade = predictor.segment_and_classify(
                scan.t1ce.path,
                scan.flair.path,
                int(scan.slice_number)
            )

            # Return results
            return JsonResponse({
                'status': 'success',
                'preview_img': preview_img,
                'prediction_img': prediction_img,
                'grade': grade
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@login_required(login_url='neuro_insight_app:login')
def save_report(request):
    """Generate and save PDF report for patient"""
    if request.method == 'POST':
        try:
            # Get form data
            patient_name = request.POST.get('patient_name')
            if not patient_name:
                return JsonResponse({'status': 'error', 'message': 'Patient name is required'})
            
            # Get the latest prediction data from session or re-run prediction
            try:
                scan = MRIScan.objects.latest('uploaded_at')
                
                # Initialize predictor
                seg_model_path = os.path.join(settings.BASE_DIR, 'models', 'seg_model.keras')
                class_model_path = os.path.join(settings.BASE_DIR, 'models', 'class_model.keras')
                predictor = BrainTumorPredictor(seg_model_path, class_model_path)
                
                # Perform prediction
                preview_img, prediction_img, grade = predictor.segment_and_classify(
                    scan.t1ce.path,
                    scan.flair.path,
                    int(scan.slice_number)
                )
                
                # Generate PDF report
                pdf_data, thumbnail_data = generate_patient_report(
                    patient_name, 
                    preview_img, 
                    prediction_img, 
                    grade
                )
                
                # Save the report to the database
                report = PatientReport(
                    user=request.user,
                    patient_name=patient_name,
                    tumor_grade=grade
                )
                
                # Save PDF file
                pdf_filename = f"{patient_name.replace(' ', '_')}_report.pdf"
                report.pdf_report.save(pdf_filename, ContentFile(pdf_data), save=False)
                
                # Save thumbnail
                thumb_filename = f"{patient_name.replace(' ', '_')}_thumbnail.png"
                report.thumbnail.save(thumb_filename, ContentFile(thumbnail_data), save=False)
                
                report.save()
                
                return JsonResponse({
                    'status': 'success',
                    'message': 'Report saved successfully',
                    'report_id': report.id
                })
                
            except MRIScan.DoesNotExist:
                return JsonResponse({'status': 'error', 'message': 'No MRI scan found. Please upload scan files first.'})
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@login_required(login_url='neuro_insight_app:login')
def dashboard(request):
    """User dashboard showing saved reports"""
    reports = PatientReport.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'neuro_insight_app/dashboard.html', {'reports': reports})

@login_required(login_url='neuro_insight_app:login')
def view_report(request, report_id):
    """View a specific PDF report"""
    report = get_object_or_404(PatientReport, id=report_id, user=request.user)
    # Serve the PDF file
    response = HttpResponse(report.pdf_report, content_type='application/pdf')
    response['Content-Disposition'] = f'inline; filename="{os.path.basename(report.pdf_report.name)}"'
    return response

@login_required(login_url='neuro_insight_app:login')
def delete_report(request, report_id):
    """Delete a specific report"""
    if request.method == 'POST':
        report = get_object_or_404(PatientReport, id=report_id, user=request.user)
        report.delete()
        return redirect('neuro_insight_app:dashboard')
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})