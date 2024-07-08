import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore UserWarnings

# Load dataset
df = pd.read_csv('diabetes.csv')

# Split dataset
X = df.drop(columns="Outcome")
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#making application class for GUI
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        #the title
        self.title("Diabetes Decoded")
        self.geometry("800x600")
        
        # Initialize frames
        self.frames = {}
        #iteration among frames
        for F in (HomeFrame, TargetVariableFrame, DataVisualizationFrame, CorrelationMatrixFrame, FeatureHistogramsFrame, BoxPlotsFrame, ModelFrame):
            page_name = F.__name__
            frame = F(parent=self, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        self.show_frame("HomeFrame")
    
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class HomeFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        label = tk.Label(self, text="Home", font=("Arial", 24))
        label.pack(pady=10)
        
        buttons = [
            ("Target Variable", "TargetVariableFrame"),
            ("Data Visualization", "DataVisualizationFrame"),
            ("Correlation Matrix", "CorrelationMatrixFrame"),
            ("Feature Histograms", "FeatureHistogramsFrame"),
            ("Box Plots", "BoxPlotsFrame"),
            ("Model", "ModelFrame")
        ]
        
        for text, frame_name in buttons:
            button = tk.Button(self, text=text, command=lambda name=frame_name: controller.show_frame(name))
            button.pack(pady=5)
            
class BackButtonFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        back_button = tk.Button(self, text="Back to Home", command=lambda: controller.show_frame("HomeFrame"))
        back_button.pack(pady=5)
        
class TargetVariableFrame(BackButtonFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        
        figure = plt.Figure(figsize=(6, 4), dpi=100)
        ax = figure.add_subplot(111)
        sns.countplot(x='Outcome', data=df, ax=ax)
        ax.set_title('Distribution of Outcome')
        
        canvas = FigureCanvasTkAgg(figure, self)
        canvas.get_tk_widget().pack(pady=10)

class DataVisualizationFrame(BackButtonFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        
        # Set the figure size
        figure = sns.pairplot(df, hue='Outcome').fig
        figure.set_size_inches(8, 6)  # Set the figure size to 8x6 inches
        
        # Create the canvas and add it to the frame
        canvas = FigureCanvasTkAgg(figure, self)
        canvas.get_tk_widget().pack(pady=10)

class CorrelationMatrixFrame(BackButtonFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        
        corr_matrix = df.corr()
        figure = plt.Figure(figsize=(8,6), dpi=100)
        ax = figure.add_subplot(111)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix')
        
        canvas = FigureCanvasTkAgg(figure, self)
        canvas.get_tk_widget().pack(pady=10)

class FeatureHistogramsFrame(BackButtonFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        
        figure, axes = plt.subplots(3, 3, figsize=(8, 6))
        axes = axes.flatten()
        df.hist(bins=20, ax=axes)
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(figure, self)
        canvas.get_tk_widget().pack(pady=10)

class BoxPlotsFrame(BackButtonFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        
        # Add the code for the box plots here
        
        # Example placeholder:
        figure = plt.Figure(figsize=(6, 4), dpi=100)
        ax = figure.add_subplot(111)
        sns.boxplot(data=df, ax=ax)
        ax.set_title('Box Plots')
        
        canvas = FigureCanvasTkAgg(figure, self)
        canvas.get_tk_widget().pack(pady=10)

class ModelFrame(BackButtonFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        
        self.result_label = None
        
        label = tk.Label(self, text="Model", font=("Arial", 24))
        label.pack(pady=10)
        
        # Create and set the model selection variable
        self.model_var = tk.StringVar(value="Logistic Regression")

        # Model selection radio buttons
        ttk.Label(self, text="Select Model:").pack(pady=2)
        ttk.Radiobutton(self, text="Logistic Regression", variable=self.model_var, value="Logistic Regression").pack(anchor=tk.W)
        ttk.Radiobutton(self, text="Decision Tree", variable=self.model_var, value="Decision Tree").pack(anchor=tk.W)

        # Create input fields for each symptom
        self.create_input_field("Pregnancies")
        self.create_input_field("Glucose")
        self.create_input_field("Blood Pressure")
        self.create_input_field("Skin Thickness")
        self.create_input_field("Insulin")
        self.create_input_field("BMI")
        self.create_input_field("Diabetes Pedigree Function")
        self.create_input_field("Age")

        # Create a button to make predictions
        predict_button = ttk.Button(self, text="Predict", command=self.predict)
        predict_button.pack(pady=5)
        
        # Button for model comparison
        compare_button = ttk.Button(self, text="Compare Models", command=self.compare_models)
        compare_button.pack(pady=5)
    
    def create_input_field(self, label_text):
        label = ttk.Label(self, text=label_text + ":")
        label.pack(pady=2)
        entry = ttk.Entry(self)
        entry.pack(pady=2)
        setattr(self, label_text.replace(" ", "_").lower() + "_entry", entry)

    def predict(self):
        # Retrieve input values
        input_values = [
            float(getattr(self, "pregnancies_entry").get()),
            float(getattr(self, "glucose_entry").get()),
            float(getattr(self, "blood_pressure_entry").get()),
            float(getattr(self, "skin_thickness_entry").get()),
            float(getattr(self, "insulin_entry").get()),
            float(getattr(self, "bmi_entry").get()),
            float(getattr(self, "diabetes_pedigree_function_entry").get()),
            float(getattr(self, "age_entry").get())
        ]

        # Select model
        model_name = self.model_var.get()
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier()

        # Train the model
        model.fit(X_train, y_train)

        # Make prediction
        prediction = model.predict([input_values])
        result = "Diabetes" if prediction[0] == 1 else "No Diabetes"

        # Show result
        if self.result_label:
            self.result_label.config(text="Prediction: " + result)
        else:
            self.result_label = tk.Label(self, text="Prediction: " + result, font=("Arial", 16))
            self.result_label.pack(pady=10)

    def compare_models(self):
        # Train Logistic Regression
        lr_model = LogisticRegression(max_iter=200)
        lr_model.fit(X_train, y_train)
        lr_y_pred = lr_model.predict(X_test)
        
        # Evaluate Logistic Regression
        lr_accuracy = accuracy_score(y_test, lr_y_pred)
        lr_precision = precision_score(y_test, lr_y_pred)
        lr_recall = recall_score(y_test, lr_y_pred)
        lr_f1 = f1_score(y_test, lr_y_pred)
        
        print("Logistic Regression:")
        print(f"Accuracy: {lr_accuracy:.4f}")
        print(f"Precision: {lr_precision:.4f}")
        print(f"Recall: {lr_recall:.4f}")
        print(f"F1 Score: {lr_f1:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, lr_y_pred)}")
        print(f"Classification Report:\n{classification_report(y_test, lr_y_pred)}")
        
        # Train Decision Tree
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        dt_y_pred = dt_model.predict(X_test)
        
        # Evaluate Decision Tree
        dt_accuracy = accuracy_score(y_test, dt_y_pred)
        dt_precision = precision_score(y_test, dt_y_pred)
        dt_recall = recall_score(y_test, dt_y_pred)
        dt_f1 = f1_score(y_test, dt_y_pred)
        
        print("\nDecision Tree:")
        print(f"Accuracy: {dt_accuracy:.4f}")
        print(f"Precision: {dt_precision:.4f}")
        print(f"Recall: {dt_recall:.4f}")
        print(f"F1 Score: {dt_f1:.4f}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, dt_y_pred)}")
        print(f"Classification Report:\n{classification_report(y_test, dt_y_pred)}")
        
        # Conclusion
        if lr_accuracy > dt_accuracy:
            print("\nConclusion: Logistic Regression performs better.")
        else:
            print("\nConclusion: Decision Tree performs better.")

if __name__ == "__main__":
    app = Application()
    app.mainloop()
