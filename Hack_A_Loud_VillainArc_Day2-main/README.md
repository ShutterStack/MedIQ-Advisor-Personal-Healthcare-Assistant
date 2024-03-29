# **MedIQ Advisor: Your Personal Healthcare Companion**
Ask Any query related to your health to our chatbot (MedIQ Advisor) and it will assist you in solving your health concern and provide valid input.
Our chatbot provides recommendations based on the risk assessment, such as self-help, seeking professional help, or contacting healthcare providers.


![Website](./MedIQ_Advisor/readme_resources/web11.png)

Application Architecture:

![Website](./MedIQ_Advisor/readme_resources/arc.png)

## Problems Identified:

- Healthcare misinformation is a significant issue, and current challenges lie in harnessing AI chatbots to deliver instant and accurate medical information to the public.
- The challenge involves the need to establish trust in AI-based chatbots as reliable sources of health information, fostering user confidence in seeking guidance on public health issues.

## Solution Proposed:
 **MedIQ Advisor: Your Personal Healthcare Companion**

- The project aims to implement a dynamic learning mechanism using machine learning algorithms to continuously update the chatbot's dataset, ensuring accurate responses to evolving health-related queries.
- Project functions with a validation mechanisms within the chatbot architecture to enhance the reliability of information, providing users with trustworthy and up-to-date health insights.

## Features:

- General Public Seeking Health Information: Individuals looking for instant and reliable information on symptoms, preventive measures, or general health queries can benefit from an AI chatbot. 

- Emergency Situations and Outbreaks: During public health emergencies or outbreaks, the chatbot can rapidly disseminate accurate information, offer guidance on protective measures, and address common medical practices. 

- Health Education Programs: Educational institutions, public health organizations, and community health programs can integrate the chatbot into health education initiatives, promoting awareness.

- Healthcare Professionals for Reference: Healthcare professionals can use the chatbot as a quick reference tool for general health information, allowing them to focus on more complex medical issues while ensuring that the information provided aligns with current medical knowledge.

- Reliability in Critical Situations : In critical healthcare scenarios, where immediate and accurate information is crucial, our chatbot demonstrates high accuracy.

- Seamless Integration: Integrating with telehealth services or healthcare providers, enabling users to transition from self-assessment to professional support seamlessly.

- Evidence-Based Algorithms: Using validated and evidence-based algorithms for accurate mental health assessment, increasing user trust in the app's results.

- Personal Mental Health Checkup: Users can use the app to regularly assess their mental health status, helping them identify changes or potential concerns over time. 

- Inter-operability with Healthcare Systems: Ensuring seamless integration with existing healthcare systems and other telemedicine platforms is essential for the chatbot to effectively collaborate with healthcare professionals.


## Screening Prediction - Model Training:

    Dataset used: survey.csv
    (combination of PHQ-9, GAD-7 datasets & other important constraints mainly targeting corporate employees).

- Step 1: Data Cleaning - Filling NA values & to drop age data out of range of 18-60.

![Website](./MedIQ_Advisor/readme_resources/sp1.png)

- Step 2: Neutralizing Data – Gender data with irregular entries gets neutralized to 3 standard entries.

![Website](./MedIQ_Advisor/readme_resources/sp2.png)

- Step 3: Data transformation – ColumnTransformer to transform data to a format.

![Website](./MedIQ_Advisor/readme_resources/sp3.png)

- Step 4: Choosing Algorithm – Using AdaBoost Algorithm, since it returns maximum accuracy.

![Website](./MedIQ_Advisor/readme_resources/sp4.png)

- Step 5: Search Technique – Using RandomizedSearchCV Search Technique.

![Website](./MedIQ_Advisor/readme_resources/sp5.png)

- Step 6: Confusion Matrix – Returning Type-1 & Type-2 Error.

![Website](./MedIQ_Advisor/readme_resources/sp6.png)

- Step 7: Plotting ROC curve.

![Website](./MedIQ_Advisor/readme_resources/sp7.png)

- Step 8: Serialization & Deserialization – Pickling abc_tuned model to model.pkl file(Byte System) & Returning screening score of some dataset entries using predict_proba function.

![Website](./MedIQ_Advisor/readme_resources/sp8.png)

## Emotion detect:

    Algorithm used: HaarCascade.

- Loading the model.

![Website](./MedIQ_Advisor/readme_resources/edc1.png)

- Loading face and recognizing emotion with the help of algorithm.
- Adding a timeout to capture 30 fps for 5 seconds.

![Website](./MedIQ_Advisor/readme_resources/edc2.png)
![Website](./MedIQ_Advisor/readme_resources/edc3.png)

- Initial page for emotion recognition.

![Website](./MedIQ_Advisor/readme_resources/ed1.jpeg)

- Successfully identifying the emotions.

![Website](./MedIQ_Advisor/readme_resources/ed2.jpeg)
![Website](./MedIQ_Advisor/readme_resources/ed3.jpeg)

## List of Questions for Prediction:

- What is your age?
- Choose your gender.
- Are you self employed?
- Do you have a family history of mental illness? 
- If you have a mental health condition, do you feel that interferes with your work?
- How many employees does your company or organization have?
- Do you work remotely (outside of an office) at least 50% of the time?
- Is your employer primarily a tech company/organization?
- Does your employer provide mental health benefits?
- Do you know the options for mental health care your employer provides?
- Has your employer ever discussed mental health as part of an employee wellness program?
- Does your employer provide resources to learn more about mental health issues and how to seek help?
- Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
- How easy is it for you to take medical leave for a mental health condition?
- Do you think that discussing a mental health issue with your employer would have negative consequences?
- Do you think that discussing a physical health issue with your employer would have negative consequences?
- Would you be willing to discuss a mental health issue with your coworkers?
- Would you be willing to discuss a mental health issue with your direct supervisors?
- Would you bring up mental health issue with a potential employer in an interview?
- Would you bring up a physical health issue with a potential employer in an interview?
- Do you feel that your employer takes mental health as seriously as physical health?
- Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?

## ChatBot
- Loading the Dataset and training-testing it

![Website](./MedIQ_Advisor/readme_resources/cb1.png)

- Using Decision Tree Algorithm 

![Website](./MedIQ_Advisor/readme_resources/cb2.png)

- Function to print and check for diseases

![Website](./MedIQ_Advisor/readme_resources/cb3.png)


- Final prediction

![Website](./MedIQ_Advisor/readme_resources/cb4.png)

## Our Team:

![Website](./MedIQ_Advisor/readme_resources/web_team.png)

## Social links:

### GitHub:

- [Himanshu Maurya](https://github.com/himanshumaurya0007)
- [Faizan Mahimkar](https://github.com/Faizan-Mahimkar)
- [Harshal Patil](https://github.com/Harshal4511)

### LinkedIn:

- [Himanshu Maurya](https://www.linkedin.com/in/himanshumaurya0007)
- [Faizan Mahimkar](https://www.linkedin.com/in/faizan-mahimkar)
- [Harshal Patil](https://www.linkedin.com/in/harshal-patil4511)
