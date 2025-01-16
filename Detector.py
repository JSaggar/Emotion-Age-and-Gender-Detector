import tkinter as tk
from tkinter import Label, Button, Entry, Frame
from tkinter import font as tkfont
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import face_recognition
from PIL import Image, ImageTk
import csv

# Load pre-trained models
# **UPDATE FILE LOCATION** 
face_exp_model = load_model("FILE LOACTION")

emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Gender and Age models 
# Verify Model Locations
gender_label_list = ['Male', 'Female']
age_label_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_protext = r"dataset/gender_deploy.prototxt"
gender_caffemodel = r"dataset/gender_net.caffemodel"
age_protext = r"dataset/age_deploy.prototxt"
age_caffemodel = r"dataset/age_net.caffemodel"
gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Initialize Tkinter window
app = tk.Tk()
app.title("Face Analysis App")
app.geometry("700x520")
app.config(bg="#f0f0f0")  # Background color

# Define fonts for different labels and buttons
label_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
button_font = tkfont.Font(family="Helvetica", size=12, weight="bold")
diagnosis_font = tkfont.Font(family="Helvetica", size=14, weight="bold")

# Label for video feed
video_label = Label(app)
video_label.pack(pady=20)

# Capture video
webcam_video_stream = cv2.VideoCapture(0)
is_running = False  # Variable to control the loop

# Initialize global variables for emotion and question flow
detected_emotion = ""
data = {"Question": [], "Answer": [], "Emotion": []}
current_question_index = 0
person_name = ""  # Default to empty string

# New set of questions with score-based responses
questions = [
    "On a scale of 1-5, how often do you feel anxious or stressed?",
    "On a scale of 1-5, how often do you have trouble sleeping?",
    "On a scale of 1-5, how often do you feel isolated or lonely?",
    "On a scale of 1-5, how often do you feel low or lack energy?",
    "On a scale of 1-5, how often do you find it hard to focus?",
    "On a scale of 1-5, how often do you experience mood swings?",
    "On a scale of 1-5, how often do you feel happy or satisfied with your life?"
]

# Function to start the analysis automatically after creating a new person
def start_analysis():
    global is_running
    if not is_running:  # Only start if not already running
        is_running = True
        analyze_frame()
    
    # Hide Create New Person button after being clicked
    new_person_button.pack_forget()

# Function to detect and display age, gender, and emotion
def analyze_frame():
    global detected_emotion  # Declare as global to be accessed in other functions
    
    if not is_running:
        return
    
    ret, current_frame = webcam_video_stream.read()
    if not ret:
        return
    
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    all_face_locations = face_recognition.face_locations(current_frame_small, model='hog')

    for index, current_face_location in enumerate(all_face_locations):
        top_pos, right_pos, bottom_pos, left_pos = [v * 4 for v in current_face_location]
        current_face_image = current_frame[top_pos:bottom_pos, left_pos:right_pos]

        # Age and Gender Prediction
        face_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender Prediction
        gender_cov_net.setInput(face_blob)
        gender_predictions = gender_cov_net.forward()
        gender = gender_label_list[gender_predictions[0].argmax()]

        # Age Prediction
        age_cov_net.setInput(face_blob)
        age_predictions = age_cov_net.forward()
        age = age_label_list[age_predictions[0].argmax()]

        # Emotion Prediction
        gray_face = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (48, 48))
        img_pixels = image.img_to_array(gray_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        exp_predictions = face_exp_model.predict(img_pixels)
        detected_emotion = emotions_label[np.argmax(exp_predictions)]  # Update global detected_emotion

        # Draw rectangle and labels
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        label = f"{gender}, {age}, {detected_emotion}"
        cv2.putText(current_frame, label, (left_pos, top_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display frame
    frame_image = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame_image)
    frame_image = ImageTk.PhotoImage(image=frame_image)
    video_label.config(image=frame_image)
    video_label.image = frame_image
    app.after(10, analyze_frame)

# Function to handle question flow and store data
def next_question():
    global detected_emotion  # Access global emotion variable
    global current_question_index  # Declare as global to modify it
    
    # Check if an answer is provided
    user_answer = answer_entry.get().strip()
    
    if not user_answer or not user_answer.isdigit() or not (1 <= int(user_answer) <= 5):
        print("Please enter a valid answer between 1 and 5.")
        return  # Do nothing if the answer is empty or out of range
    
    # Store the answer and emotion
    data["Question"].append(questions[current_question_index])
    data["Answer"].append(user_answer)
    data["Emotion"].append(detected_emotion)
    
    # Clear the answer entry
    answer_entry.delete(0, tk.END)
    
    # Move to next question
    current_question_index += 1
    
    if current_question_index < len(questions):
        question_label.config(text=questions[current_question_index])
    else:
        print("All questions answered, calculating diagnosis.")
        calculate_diagnosis()  # Call diagnosis calculation
        save_data_to_csv()
        # Hide Next Question button after answering all questions
        next_button.pack_forget()

# Function to calculate diagnosis
def calculate_diagnosis():
    total_score = sum(int(answer) for answer in data["Answer"])
    print(f"Total Score: {total_score}")  # Debugging: Check the score calculation

    if 7 <= total_score <= 14:
        diagnosis = "Stable mental health status. Continue with positive mental well-being practices."
    elif 15 <= total_score <= 21:
        diagnosis = "Moderate distress. Consider self-care practices; monitor feelings. Professional support may be beneficial if feelings persist."
    elif 22 <= total_score <= 28:
        diagnosis = "High distress. Frequent anxiety or low mood. Consider seeking support from a counselor or mental health professional."
    elif 29 <= total_score <= 35:
        diagnosis = "Severe distress. Strongly recommended to reach out to a mental health professional for support."
    else:
        diagnosis = "Score out of expected range. Please check responses."
    
    print(f"Diagnosis: {diagnosis}")
    result_label.config(text=f"Diagnosis: {diagnosis}", font=diagnosis_font)
    result_label.pack(pady=20)  # Show diagnosis result

# Save collected data to CSV
def save_data_to_csv():
    global person_name
    with open('mental_health_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Leave a blank line before new user's data
        writer.writerow([])  
        
        writer.writerow([person_name])  # Write person's name
        writer.writerow(["Question", "Answer", "Emotion"])
        
        for i in range(len(data["Question"])):
            writer.writerow([data["Question"][i], data["Answer"][i], data["Emotion"][i]])

# Function to create a new entry for a new person
def create_new_person():
    global person_name
    person_name = name_entry.get().strip()
    if not person_name:
        print("Please enter a name.")
        return
    print(f"New person: {person_name}")
    data["Question"].clear()
    data["Answer"].clear()
    data["Emotion"].clear()
    global current_question_index
    current_question_index = 0
    question_label.config(text=questions[current_question_index])

    # Hide Name Entry and 'Create New Person' Button after entering name
    name_label.pack_forget()
    name_entry.pack_forget()
    new_person_button.pack_forget()
    # Start the analysis automatically
    start_analysis()
    
    # Show question label and answer entry
    question_label.pack(pady=20)
    answer_entry.pack(pady=10)
    next_button.pack(pady=10)

# UI Components for name entry
name_label = Label(app, text="Enter your name:", font=label_font)
name_label.pack(pady=10)

name_entry = Entry(app, font=("Helvetica", 12))
name_entry.pack(pady=10)

new_person_button = Button(app, text="Create New Person", command=create_new_person, font=button_font, bg="#4CAF50", fg="white", relief="raised")
new_person_button.pack(pady=10)

# UI Components for the questions
question_label = Label(app, text="Enter your name to begin:", font=label_font, wraplength=400)
question_label.pack_forget()

answer_entry = Entry(app, font=("Helvetica", 12))
answer_entry.pack_forget()

next_button = Button(app, text="Next Question", command=next_question, font=button_font, bg="#2196F3", fg="white", relief="raised")
next_button.pack_forget()

result_label = Label(app, text="", font=diagnosis_font, wraplength=400, bg="#FFEB3B", fg="black", relief="solid", padx=10, pady=10)
result_label.pack_forget()

app.mainloop()
