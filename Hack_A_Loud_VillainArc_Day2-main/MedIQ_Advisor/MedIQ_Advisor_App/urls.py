from django.contrib import admin
from django.urls import path
from MedIQ_Advisor_App import views
from . import views

urlpatterns = [
    path("", views.index_function, name='index'),
    path("sign_in", views.sign_in_function, name='sign_in'),
    path("sign_up", views.sign_up_function, name='sign_up'),
    path("forgot_password", views.forgot_password_function, name='forgot_password'),
    path("home", views.home_function, name='home'),
    path("contact", views.contact_function, name='contact'),
    path("emotion_questionnaire", views.emotion_questionnaire_function, name='emotion_questionnaire'),
    path("heart_health", views.heart_health_function, name='heart_health'),
    path("diabetes", views.diabetes_function, name='diabetes'),
    path("brain_tumor", views.brain_tumor_function, name='brain_tumor'),
]