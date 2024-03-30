from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
import cv2
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import numpy as np
import time
import re  # Import regular expression module for username and email validation

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///MedIQ_Advisor_flask.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'Medical_Health_Advisor'
db = SQLAlchemy(app)

# ********************************** DB User table **********************************
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firstName = db.Column(db.String(15), nullable=False)
    lastName = db.Column(db.String(15), nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

# ********************************** DB Contact Us table **********************************
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)

# ********************************** emotion detection **********************************
# Load model
emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}
json_file = open('static/emotion_detection/emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("static/emotion_detection/emotion_model1.h5")

# Load face
try:
    face_cascade = cv2.CascadeClassifier('static/emotion_detection/haarcascade_frontalface_default.xml')
except Exception:
    print("Error loading cascade classifiers")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.start_time = time.time()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 5:
            return None

        success, frame = self.video.read()
        if not success:
            return None

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = img_gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)

            label_position = (x, y)
            cv2.putText(frame, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


# ********************************** Creates all DB tables **********************************
with app.app_context():
    db.create_all()
    

# ********************************** Page routing or navigation ********************************** #
@app.route("/")
def index():
    return render_template('index.html')


# ********************************** Due to emotion detection **********************************
def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Define regular expression patterns for name, username, email, and password validation
name_pattern = re.compile(r'^[a-zA-Z]{1,15}$')
username_pattern = re.compile(r'^[a-z0-9_.]{6,}$')
email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@(?:gmail|yahoo|outlook)\.com$')
password_pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')

@app.route('/sign_up',  methods=['GET', 'POST']) # Accept both GET and POST requests
def sign_up():
    if request.method == 'POST':
        firstName = request.form['firstName']
        lastName = request.form['lastName']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if the first name and last name meet the criteria
        if not name_pattern.match(firstName) or not name_pattern.match(lastName):
            flash('First name and last name must be between 1 and 15 characters long and contain only letters.', 'error')
            return redirect('/sign_up')

        # Check if the username meets the criteria
        if not username_pattern.match(username):
            flash('Username must contain at least 6 characters and can only contain letters (in small-case only), numbers (0 to 9), underscore (_), and period (.)', 'error')
            return redirect('/sign_up')

        # Check if the email meets the criteria
        if not email_pattern.match(email):
            flash('Email must be in a valid format (@gmail.com, @yahoo.com, @outlook.com, etc.)', 'error')
            return redirect('/sign_up')

        # Check if the password meets the criteria
        if not password_pattern.match(password):
            flash('Password must contain at least 8 characters, 1 capital letter, 1 small letter, and 1 symbol.', 'error')
            return redirect('/sign_up')

        # Check if the username or email already exists
        if User.query.filter_by(username=username).first() is not None:
            flash('User already exists! Choose a different username.', 'error')
            return redirect('/sign_up')
        elif User.query.filter_by(email=email).first() is not None:
            flash('Email already exists! Use a different email address.', 'error')
            return redirect('/sign_up')

        # If all validation passes, create a new user
        new_user = User(firstName=firstName, lastName=lastName, username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Now, user can sign in.', 'success')
        return redirect('/sign_up')

    # If it's a GET request, just render the sign_up form
    return render_template('sign_up.html')


@app.route('/sign_in', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Find user by username
        user = User.query.filter_by(username=username).first()

        if user:
            # Check if the password matches
            if user.password == password:
                flash('sign_in successful!', 'success')
                return redirect('/home')
            else:
                flash('Incorrect password. Please try again.', 'error')
                return redirect('/sign_in')  # Redirect after flashing error message
        else:
            flash('User not found. Please sign_up first.', 'error')
            return redirect('/sign_up')

    return render_template('sign_in.html')


@app.route('/forgot_password', methods=['GET', 'POST'])
def update_password():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        new_password = request.form['password']

        # Check if the password meets the criteria
        if not password_pattern.match(new_password):
            flash('Password must contain at least 8 characters, 1 capital letter, 1 small letter, and 1 symbol.', 'error')
            return redirect('/forgot_password')

        # Check if the username and email match the user
        user = User.query.filter_by(username=username, email=email).first()

        if user:
            # Update the password
            user.password = new_password
            db.session.commit()
            flash('Password updated successfully!', 'success')
        else:
            flash('Username or email did not match. Check either of them before trying again.', 'error')
    
    return render_template('forgot_password.html')


@app.route("/contact_us", methods=["GET", "POST"])
def contact_us():
    if request.method == "POST":
        name = request.form["name"]
        phone = request.form["phone"]
        email = request.form["email"]
        description = request.form["description"]

        new_message = ContactMessage(name=name, phone=phone, email=email, description=description)
        db.session.add(new_message)
        db.session.commit()

        flash('Your message has been sent successfully!", "success')

        return redirect(url_for("contact_us"))
    
    return render_template("contact_us.html")

@app.route("/about_us")
def about_us():
    return render_template("about_us.html")

@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/emotion_detection")
def camera():
    return render_template("emotion_detection.html")

@app.route("/emotion_questionnaire")
def emotion_questionnaire():
    return render_template("emotion_questionnaire.html")

# ********************************** Due to emotion detection **********************************
@app.route('/video_feed_emotion')
def video_feed_emotion():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/voice_assistant")
def voice_assistant():
    return render_template("voice_assistant.html")

if __name__ == "__main__":
    app.run(debug=True)