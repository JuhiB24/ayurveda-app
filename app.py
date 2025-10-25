import pandas as pd
import joblib
import time
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from werkzeug.serving import WSGIRequestHandler  # For HTTP/1.1 adjustment

# Set HTTP/1.1 to ensure proper connection closing
WSGIRequestHandler.protocol_version = "HTTP/1.1"

# Initialize Flask app and SQLAlchemy
app = Flask(__name__)

# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(15), nullable=False)

    def __repr__(self):
        return f"<User {self.first_name} {self.last_name}>"

# Load the dataset
df = pd.read_csv('D:/ayurveda_app_SDE/ayurvedic_diseases_and_treatments_cleaned (1).csv')

# Normalize symptoms to lowercase
df['Symptoms'] = df['Symptoms'].str.lower()

# Prepare features and target
X = df[['Disease', 'Symptoms']]
y = df['Ayurvedic Treatment']

# Encode categorical features
X_encoded = pd.get_dummies(X, columns=['Disease', 'Symptoms'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Model Loading Logic Optimization (Only load once at the start of the app)
model = None

def load_model():
    global model
    if model is None:
        try:
            model = joblib.load('model.pkl')
            print("Loaded pre-trained model.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    return model

# Function to predict disease and treatment
def predict_treatment(user_symptoms):
    user_symptoms_set = set([symptom.lower() for symptom in user_symptoms])
    df['Symptom_Match'] = df['Symptoms'].apply(lambda x: user_symptoms_set.intersection(set(x.split(", "))))

    potential_matches = df[df['Symptom_Match'].apply(len) > 0]

    if not potential_matches.empty:
        result = []
        for index, row in potential_matches.iterrows():
            result.append({
                "Disease": row['Disease'],
                "Ayurvedic Treatment": row['Ayurvedic Treatment'],
                "Matched Symptoms": ', '.join(row['Symptom_Match'])
            })
        return result
    else:
        return [{"message": "No related diseases found. Please refine your input."}]

# Routes for handling login, registration, and prediction
@app.route('/')
def index():
    return redirect(url_for('login'))  # Redirect to the login page

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']  # Currently no password check; add logic if needed
        user = User.query.filter_by(email=email).first()
        if user:
            return redirect(url_for('extra'))  # Redirect to the extra.html page after login
        else:
            return "User not found. Please register first."
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first-name', '').strip()
        last_name = request.form.get('last-name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()

        # Ensure all required fields are provided
        if not all([first_name, last_name, email, phone]):
            return "All fields are required.", 400

        # Check if email already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "This email is already registered. Please use a different email."

        # Add user to the database
        new_user = User(
            first_name=first_name, last_name=last_name, email=email, phone=phone
        )
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/extra')
def extra():
    return render_template('extra.html')  # Render extra.html page after login

@app.route('/home')
def home():
    return render_template('home.html')  # Main home page after clicking 'Check Yourself'

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['symptoms']
    user_symptoms = [symptom.strip() for symptom in user_input.split(",")]

    # Load model only if not already loaded
    model = load_model()
    if model:
        prediction_results = predict_treatment(user_symptoms)
    else:
        prediction_results = "Model failed to load. Please try again later."

    return render_template('home.html', prediction=prediction_results)

if __name__ == '__main__':
    with app.app_context():
        # Drop the existing 'user' table and recreate it with the updated schema
        db.drop_all()  # This will drop all tables, including the old user table
        db.create_all()  # Recreate all tables, including the updated user table
    app.run(debug=True, port=5001)  # Changed port to 5001 to avoid conflicts
