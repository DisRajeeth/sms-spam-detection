import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import CalibratedClassifierCV
import tkinter as tk
from tkinter import messagebox, font

sms = pd.read_csv('C:/Python Programs/sms-spam-detection/spam.csv', encoding='utf-8', encoding_errors='ignore')
sms_train = sms[:5575]

classifier = PassiveAggressiveClassifier(max_iter=1000, random_state=2786)
vectorizer = HashingVectorizer()
vectorize_sms = vectorizer.fit_transform(sms_train.v2)
classifier.fit(vectorize_sms, sms_train.v1)

calibrated_clf = CalibratedClassifierCV(classifier, cv=5)
calibrated_clf.fit(vectorize_sms, sms_train.v1)

def classify_sms():
    a = sms_input.get()
    vectorize_t_sms = vectorizer.transform([a])
    prediction = classifier.predict(vectorize_t_sms)[0]
    probability = calibrated_clf.predict_proba(vectorize_t_sms)
    
    if prediction == 'spam':
        result = "Spam"
    else:
        result = "Not Spam"

    probability_message = f"Probability of being Spam: {probability[0][1]:.2f}\nProbability of being Not Spam: {probability[0][0]:.2f}"
    messagebox.showinfo("Result", f"Message: {result}\n{probability_message}")

root = tk.Tk()
root.title("SMS Spam Detection")
root.geometry("400x300")
root.config(bg="#f0f0f0")

custom_font = font.Font(family="Times New Roman", size=12)

label = tk.Label(root, text="Enter your SMS:", bg="#f0f0f0", font=custom_font)
label.pack(pady=20)

sms_input = tk.Entry(root, width=40, font=custom_font, bd=2, relief="solid")
sms_input.pack(pady=10)

submit_button = tk.Button(root, text="Classify SMS", command=classify_sms, font=custom_font, bg="#4CAF50", fg="white", bd=0)
submit_button.pack(pady=20) 

footer_label = tk.Label(root, text="SMS Spam Detection Tool", bg="#f0f0f0", font=custom_font)
footer_label.pack(side="bottom", pady=10)

root.mainloop()