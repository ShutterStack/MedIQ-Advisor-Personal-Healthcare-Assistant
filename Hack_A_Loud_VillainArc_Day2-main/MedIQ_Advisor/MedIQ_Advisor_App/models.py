from django.db import models

# makemigrations(meaning) => create changes and store in a file
# migrate(meaning) => apply the pending chamges created by makemigrations

# # Create your models here.
class Contact(models.Model):
    # id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    email = models.CharField(max_length=100)
    desc = models.TextField()
    date = models.DateField()

    def __str__(self):
        return self.name

class Sign_up(models.Model):
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    email = models.EmailField()
    username = models.CharField(max_length=50)
    security_question = models.CharField(max_length=100)
    security_question_answer = models.CharField(max_length=255)
    password = models.CharField(max_length=50)

    def __str__(self):
        return self.first_name

class Mental_Health_Survey(models.Model):
    id = models.AutoField(primary_key=True)
    age = models.IntegerField()
    gender = models.CharField(max_length=20)
    self_employed = models.CharField(max_length=5)
    family_history = models.CharField(max_length=5)
    mental_health_interference = models.CharField(max_length=10)
    company_size = models.CharField(max_length=20)
    remote_work = models.CharField(max_length=5)
    tech_company = models.CharField(max_length=5)
    mental_health_benefits = models.CharField(max_length=5)
    know_mental_health_care = models.CharField(max_length=10)
    discussed_mental_health = models.CharField(max_length=10)
    resources_learn_mental_health = models.CharField(max_length=10)
    anonymity_protected = models.CharField(max_length=10)
    medical_leave = models.CharField(max_length=20)
    negative_consequences_mental_health = models.CharField(max_length=10)
    negative_consequences_physical_health = models.CharField(max_length=10)
    discuss_with_coworkers = models.CharField(max_length=10)
    discuss_with_supervisors = models.CharField(max_length=10)
    bring_up_in_interview_mental_health = models.CharField(max_length=10)
    bring_up_in_interview_physical_health = models.CharField(max_length=10)
    employer_takes_mental_health_seriously = models.CharField(max_length=10)
    observed_negative_consequences = models.CharField(max_length=5)

    def __str__(self):
        return str(self.id)
    
class Heart_Health_Survey(models.Model):
    id = models.AutoField(primary_key=True)
    age = models.IntegerField()
    gender = models.CharField(max_length=20)
    chest_pain = models.CharField(max_length=20)
    resting_blood_pressure = models.IntegerField()
    serum_cholesterol = models.IntegerField()
    high_fasting_blood_sugar = models.CharField(max_length=5)
    resting_ecg = models.CharField(max_length=5)
    max_heart_rate = models.CharField(max_length=20)
    exercise_induced_angina = models.CharField(max_length=5)
    st_depression = models.CharField(max_length=20)
    peak_exercise_st_slope = models.CharField(max_length=20)
    colored_vessels = models.CharField(max_length=5)
    thalassemia_type = models.CharField(max_length=20)

    def __str__(self):
        return str(self.id)
    
class Diabetes_Survey(models.Model):
    id = models.AutoField(primary_key=True)
    pregnancies = models.IntegerField()
    glucose_level = models.IntegerField()
    blood_pressure = models.CharField(max_length=20)
    skin_thickness = models.IntegerField()
    insulin_level = models.IntegerField()
    bmi = models.FloatField()
    diabetes_pedigree = models.FloatField()
    age = models.IntegerField()

    def __str__(self):
        return str(self.id)
    
class Brain_Tumor_Survey(models.Model):
    id = models.AutoField(primary_key=True)
    brain_tumor_diagnosis = models.CharField(max_length=3, choices=[("yes", "Yes"), ("no", "No")])
    mean_radius = models.CharField(max_length=255)
    mean_texture = models.CharField(max_length=255)
    mean_perimeter = models.CharField(max_length=255)
    mean_area = models.CharField(max_length=255)
    smoothness = models.CharField(max_length=255)
    mean_compactness = models.CharField(max_length=255)
    concavity = models.CharField(max_length=255)
    concave_points = models.CharField(max_length=255)
    symmetry = models.CharField(max_length=255)
    mean_fractal_dimension = models.CharField(max_length=255)
    se_radius = models.CharField(max_length=255)
    se_texture = models.CharField(max_length=255)
    se_perimeter = models.CharField(max_length=255)
    se_area = models.CharField(max_length=255)
    se_smoothness = models.CharField(max_length=255)
    se_compactness = models.CharField(max_length=255)
    se_concavity = models.CharField(max_length=255)
    se_concave_points = models.CharField(max_length=255)
    se_symmetry = models.CharField(max_length=255)
    se_fractal_dimension = models.CharField(max_length=255)
    worst_radius = models.CharField(max_length=255)
    worst_texture = models.CharField(max_length=255)
    worst_perimeter = models.CharField(max_length=255)
    worst_area = models.CharField(max_length=255)
    worst_smoothness = models.CharField(max_length=255)
    worst_compactness = models.CharField(max_length=255)
    worst_concavity = models.CharField(max_length=255)
    worst_concave_points = models.CharField(max_length=255)
    worst_symmetry = models.CharField(max_length=255)
    worst_fractal_dimension = models.CharField(max_length=255)

    def __str__(self):
        return str(self.id)