# neuro_insight_app/urls.py
from django.urls import path
from . import views

app_name = 'neuro_insight_app'
urlpatterns = [
    path('', views.home, name='home'),
    path('our_team/', views.our_team, name='our_team'),
    path('home_screen/', views.home_screen, name='home_screen'),
    path('upload/', views.upload_mri, name='upload_mri'),
    path('predict/', views.predict, name='predict'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('save-report/', views.save_report, name='save_report'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('report/<int:report_id>/', views.view_report, name='view_report'),
    path('report/<int:report_id>/delete/', views.delete_report, name='delete_report'),
]