from django.contrib import admin
from django.urls import path
from VR_Assistant_App import views
from . import views

urlpatterns = [
    path('', views.voice_assistant_function, name='voice_assistant_function'),
]