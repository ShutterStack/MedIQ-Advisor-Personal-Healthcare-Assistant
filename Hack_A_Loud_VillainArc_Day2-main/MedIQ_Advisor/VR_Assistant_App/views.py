from django.shortcuts import render

# Create your views here.
def voice_assistant_function(request):
    return render(request, 'VR_Assistant_App_Template/voice_assistant.html')