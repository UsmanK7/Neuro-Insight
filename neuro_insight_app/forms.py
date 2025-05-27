# neuro_insight_app/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import MRIScan

class MRIScanForm(forms.ModelForm):
    class Meta:
        model = MRIScan
        fields = ['t1ce', 'flair', 'slice_number']
        widgets = {
            't1ce': forms.FileInput(attrs={'class': 'form-control', 'accept': '.nii,.nii.gz'}),
            'flair': forms.FileInput(attrs={'class': 'form-control', 'accept': '.nii,.nii.gz'}),
            'slice_number': forms.NumberInput(attrs={'class': 'form-control', 'min': '0'}),
        }
        labels = {
            't1ce': 'T1CE MRI (.nii)',
            'flair': 'FLAIR MRI (.nii)',
            'slice_number': 'Slice Number',
        }

class SignUpForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Enter your email'})
    )

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Choose a username'}),
        }
    
    def __init__(self, *args, **kwargs):
        super(SignUpForm, self).__init__(*args, **kwargs)
        # Add Bootstrap classes to form fields
        self.fields['password1'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Create a password'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Confirm your password'})

class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super(LoginForm, self).__init__(*args, **kwargs)
        # Add Bootstrap classes to form fields
        self.fields['username'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Enter your username'})
        self.fields['password'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Enter your password'})

class PatientNameForm(forms.Form):
    patient_name = forms.CharField(
        max_length=100, 
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter patient name'})
    )