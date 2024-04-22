from flask import Flask, render_template, request, Response, flash, url_for
from flask_sqlalchemy import SQLAlchemy
import cv2
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import numpy as np
import time
import os
import pickle
from sklearn.preprocessing import LabelEncoder
import random  # Import random module here
from flask import redirect
import re  # Import regular expression module for username and email validation
import smtplib
from email.message import EmailMessage


# Turn off oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__, template_folder='templates')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///MedIQ_Advisor_flask.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'Medical_Health_Advisor'
db = SQLAlchemy(app)


# ********************************** DB User table **********************************
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullName = db.Column(db.String(15), nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    mobile_no = db.Column(db.String(10), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    emergency_contact = db.Column(db.String(10), nullable=False)
    concern = db.Column(db.Text, nullable=True)  # Assuming concern can be optional
    password = db.Column(db.String(80), nullable=False)


# ********************************** DB Contact table **********************************
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)


# ********************************** emotion detection **********************************
# Load model architecture from JSON file
emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}
json_file = open('static/emotion_detection/emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# Load weights into the model
classifier.load_weights("static/emotion_detection/emotion_model1.h5")

# Load face cascade
try:
    face_cascade = cv2.CascadeClassifier('static/emotion_detection/haarcascade_frontalface_default.xml')
except Exception as e:
    print("Error loading cascade classifiers:", e)


# ********************************** Due to emotion detection **********************************
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

# Load screening model
with open('static/models/screening/model.sav', 'rb') as file:
    loaded_model = pickle.load(file)

# ********************************** Due to emotion detection **********************************
def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# ********************************** Creates all DB tables **********************************
with app.app_context():
    db.create_all()


# ********************************** Page routing or navigation ********************************** #
@app.route('/')
def index():
    return render_template('index.html')


# Define regular expression patterns for name, username, email, and password validation
name_pattern = re.compile(r'^[a-zA-Z\s]{1,40}$')
username_pattern = re.compile(r'^[a-z0-9_.]{6,}$')
email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@(?:gmail|yahoo|outlook)\.com$')
mobile_no_pattern = re.compile(r'^\d{10}$')
password_pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')


# Sign up page
@app.route('/signup',  methods=['GET', 'POST']) # Accept both GET and POST requests
def signup():
    if request.method == 'POST':
        fullName = request.form['fullName']
        gender = request.form['gender']
        age = request.form['age']
        username = request.form['username']
        email = request.form['email']
        mobile_no = request.form['mobile_no']
        city = request.form['city']
        emergency_contact = request.form['emergency_contact']
        concern = request.form['concern']
        password = request.form['password']

        # Check if the full name meets the criteria
        if not name_pattern.match(fullName):
            flash('Full name must be between 1 and 40 characters long and contain only letters.', 'error')
            return redirect('/signup')
        
        # Check if the gender is valid
        if gender not in ['male', 'female']:
            flash('Gender must be either "male" or "female".', 'error')
            return redirect('/signup')
        
        # Check if the age is valid
        try:
            age = int(age)
            if age < 18 or age > 100:  # Assuming valid age range is between 18 and 100
                raise ValueError
        except ValueError:
            flash('Age must be a number between 18 and 100.', 'error')
            return redirect('/signup')
        
        # Check if the username meets the criteria
        if not username_pattern.match(username):
            flash('Username must contain at least 6 characters and can only contain letters (in small-case only), numbers (0 to 9), underscore (_), and period (.)', 'error')
            return redirect('/signup')
        
         # Check if the email meets the criteria
        if not email_pattern.match(email):
            flash('Email must be in a valid format (@gmail.com, @yahoo.com, @outlook.com, etc.)', 'error')
            return redirect('/signup')
        
        # Check if the mobile number is valid
        if not mobile_no_pattern.match(mobile_no):
            flash('Mobile number must be a 10-digit number.', 'error')
            return redirect('/signup')
        
        # Check if the city is provided and meets the criteria
        if not city:
            flash('City field is required.', 'error')
            return redirect('/signup')
        
        # Check if the emergency contact is valid
        if not mobile_no_pattern.match(emergency_contact):
            flash('Emergency contact must be a 10-digit number.', 'error')
            return redirect('/signup')
        
        # Check if the concern field meets the criteria (if provided)
        if concern and len(concern) > 200:  # Assuming maximum length of concern is 200 characters
            flash('Concern must be less than or equal to 200 characters.', 'error')
            return redirect('/signup')
       
        # Check if the password meets the criteria
        if not password_pattern.match(password):
            flash('Password must contain at least 8 characters, 1 capital letter, 1 small letter, and 1 symbol.', 'error')
            return redirect('/signup')

        # Check if the username or email already exists
        if User.query.filter_by(username=username).first() is not None:
            flash('User already exists! Choose a different username.', 'error')
            return redirect('/signup')
        elif User.query.filter_by(email=email).first() is not None:
            flash('Email already exists! Use a different email address.', 'error')
            return redirect('/signup')

        # If all validation passes, create a new user
        new_user = User(fullName=fullName, gender=gender, age=age, username=username, email=email, mobile_no=mobile_no, city=city, emergency_contact=emergency_contact, concern=concern, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Now, you can sign in.', 'success')
        return redirect('/signup')

    # If it's a GET request, just render the signup form
    return render_template('signup.html')


# Sign in page 
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Find user by username
        user = User.query.filter_by(username=username).first()

        if user:
            # Check if the password matches
            if user.password == password:
                flash('Sign-in successful!', 'success')
                return redirect('/home')
            else:
                flash('Incorrect password. Please try again.', 'error')
                return redirect('/signin')  # Redirect after flashing error message
        else:
            flash('User not found. Please sign up first.', 'error')
            return redirect('/signup')

    return render_template('signin.html')


# Function to send email for password reset
def send_password_reset_email(username, email, new_password):
    # Email content
    subject = "Password Reset Request"
    body = f"Hello {username},\n\nYour password has been successfully reset. Your new password is: {new_password}\n\nIf you didn't request this change, please contact us immediately."
    to = email

    # Email configuration
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to

    user = "himanshu.m1802@gmail.com"  # Update with your email
    msg['from'] = user
    password = "anshu1802"  # Update with your email password

    # Sending the email
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)
    server.quit()

# Update the update_password() function to include sending email
@app.route('/forgotpassword', methods=['GET', 'POST'])
def update_password():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        new_password = request.form['password']

        # Check if the username and email match the user
        user = User.query.filter_by(username=username, email=email).first()

        if user:
            # Update the password
            user.password = new_password
            db.session.commit()
            # Send password reset email
            send_password_reset_email(username, email, new_password)
            flash('Password updated successfully! Check your email for confirmation.', 'success')
        else:
            flash('Username or email did not match. Check either of them before trying again.', 'error')
    
    return render_template('forgotpassword.html')

# Forgot password page
# @app.route('/forgotpassword', methods=['GET', 'POST'])
# def update_password():
#     if request.method == 'POST':
#         username = request.form['username']
#         email = request.form['email']
#         new_password = request.form['password']

#         # Check if the password meets the criteria
#         if not password_pattern.match(new_password):
#             flash('Password must contain at least 8 characters, 1 capital letter, 1 small letter, and 1 symbol.', 'error')
#             return redirect('/forgotpassword')

#         # Check if the username and email match the user
#         user = User.query.filter_by(username=username, email=email).first()

#         if user:
#             # Update the password
#             user.password = new_password
#             db.session.commit()
#             flash('Password updated successfully!', 'success')
#         else:
#             flash('Username or email did not match. Check either of them before trying again.', 'error')
    
#     return render_template('forgotpassword.html')


# Home page
@app.route('/home', methods=['POST', 'GET'])
def home():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        subject = request.form["subject"]
        message = request.form["message"]

        new_message = ContactMessage(name=name, email=email, subject=subject, message=message)
        db.session.add(new_message)
        db.session.commit()

        flash('Your message has been sent successfully!', 'success')

        return redirect("/home")

    return render_template('home.html')


@app.route('/video_feed_emotion')
def video_feed_emotion():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


# Emotion detection page
@app.route('/emotiondetection')
def emotiondetection():
    return render_template('emotiondetection.html')


# Define the route to handle form submission
@app.route('/submit_form', methods=['POST', 'GET'])
def submit_form():
    # Get the random number passed from the JavaScript function
    random_number = float(request.form['randomNumber'])

    # Determine the route based on the random number
    if 0.0 <= random_number < 0.1:
        return render_template('emotion_questionnaire_response1_1.html', random_number=random_number)
    elif 0.1 <= random_number < 0.2:
        return render_template('emotion_questionnaire_response1_2.html', random_number=random_number)
    elif 0.2 <= random_number < 0.3:
        return render_template('emotion_questionnaire_response1_3.html', random_number=random_number)
    elif 0.3 <= random_number < 0.4:
        return render_template('emotion_questionnaire_response1_4.html', random_number=random_number)
    elif 0.4 <= random_number < 0.5:
        return render_template('emotion_questionnaire_response1_5.html', random_number=random_number)
    elif 0.5 <= random_number < 0.6:
        return render_template('emotion_questionnaire_response2_1.html', random_number=random_number)
    elif 0.6 <= random_number < 0.7:
        return render_template('emotion_questionnaire_response2_2.html', random_number=random_number)
    elif 0.7 <= random_number < 0.8:
        return render_template('emotion_questionnaire_response2_3.html', random_number=random_number)
    elif 0.8 <= random_number < 0.9:
        return render_template('emotion_questionnaire_response2_4.html', random_number=random_number)
    else:
        return render_template('emotion_questionnaire_response2_5.html', random_number=random_number)


@app.route('/emotion_questionnaire_response1_2', methods=['POST', 'GET'])
def emotion_questionnaire_response1_2():
    return render_template('emotion_questionnaire_response1_2.html')

@app.route('/emotion_questionnaire_response1_3', methods=['POST', 'GET'])
def emotion_questionnaire_response1_3():
    return render_template('emotion_questionnaire_response1_3.html')

@app.route('/emotion_questionnaire_response1_4', methods=['POST', 'GET'])
def emotion_questionnaire_response1_4():
    return render_template('emotion_questionnaire_response1_4.html')

@app.route('/emotion_questionnaire_response1_5', methods=['POST', 'GET'])
def emotion_questionnaire_response1_5():
    return render_template('emotion_questionnaire_response1_5.html')

@app.route('/emotion_questionnaire_response2_1', methods=['POST', 'GET'])
def emotion_questionnaire_response2_1():
    return render_template('emotion_questionnaire_response2_1.html')

@app.route('/emotion_questionnaire_response2_2', methods=['POST', 'GET'])
def emotion_questionnaire_response2_2():
    return render_template('emotion_questionnaire_response2_2.html')

@app.route('/emotion_questionnaire_response2_3', methods=['POST', 'GET'])
def emotion_questionnaire_response2_3():
    return render_template('emotion_questionnaire_response2_3.html')

@app.route('/emotion_questionnaire_response2_4', methods=['POST', 'GET'])
def emotion_questionnaire_response2_4():
    return render_template('emotion_questionnaire_response2_4.html')

@app.route('/emotion_questionnaire_response2_5', methods=['POST', 'GET'])
def emotion_questionnaire_response2_5():
    return render_template('emotion_questionnaire_response2_5.html')


@app.route('/emotion_questionnaire', methods=['POST','GET'])
def emotion_questionnaire():
    return render_template('emotion_questionnaire.html')
   
# Chatbot page
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)