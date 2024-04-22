import os
import webbrowser
import eel

from engine.features import *
from engine.command import *

def start():
    
    eel.init("www")

    # Open the app in the default web browser
    webbrowser.open("http://localhost:8000/index.html")

    eel.start('index.html', mode=None, host='localhost', block=True)
