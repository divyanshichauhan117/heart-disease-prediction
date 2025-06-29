from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cross_val_score
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store model and scaler
model = None
scaler = None
model_performance = {}

def load_and_train_model():
    """Load data and train the best performing model"""
    global model, scaler, model_performance
    
    # Load the heart disease dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        print("✅ Dataset loaded successfully!")
    except:
        print("❌ Error loading data. Creating sample dataset...")
        # Create sample data if URL fails
        np.random.seed(42)
        n_samples = 300
        df = pd.DataFrame({
            'age': np.random.randint(30, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(120, 400, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(80, 200, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(1, 4, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
    
    # Preprocess data
    df = df.dropna()
    df['target'] = (df['target'] > 0).astype(int)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models and select the best one
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    best_accuracy = 0
    best_model_name = ""
    
    for name, mdl in models.items():
        mdl.fit(X_train_scaled, y_train)
        y_pred = mdl.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(mdl, X_train_scaled, y_train, cv=5)
        
        model_performance[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            model = mdl
    
    print(f"✅ Best model trained: {best_model_name} with accuracy: {best_accuracy:.4f}")
    return best_model_name, best_accuracy

def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, 
                         thalach, exang, oldpeak, slope, ca, thal):
    """Predict heart disease for given patient data"""
    global model, scaler
    
    if model is None or scaler is None:
        return None, None, "Model not loaded"
    
    try:
        # Create feature array
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                             thalach, exang, oldpeak, slope, ca, thal]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        return int(prediction), float(probability), "Success"
    except Exception as e:
        return None, None, str(e)

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html', model_performance=model_performance)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get data from form
        data = request.get_json()
        
        # Extract features
        age = float(data['age'])
        sex = int(data['sex'])
        cp = int(data['cp'])
        trestbps = float(data['trestbps'])
        chol = float(data['chol'])
        fbs = int(data['fbs'])
        restecg = int(data['restecg'])
        thalach = float(data['thalach'])
        exang = int(data['exang'])
        oldpeak = float(data['oldpeak'])
        slope = int(data['slope'])
        ca = int(data['ca'])
        thal = int(data['thal'])
        
        # Make prediction
        prediction, probability, status = predict_heart_disease(
            age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal
        )
        
        if status != "Success":
            return jsonify({
                'success': False,
                'error': status
            })
        
        # Prepare response
        result = {
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'risk_level': 'High Risk' if probability > 0.7 else 'Medium Risk' if probability > 0.3 else 'Low Risk',
            'message': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model_info')
def model_info():
    """Return model performance information"""
    return jsonify(model_performance)

if __name__ == '__main__':
    # Load and train model on startup
    print("🚀 Starting Heart Disease Prediction App...")
    print("📊 Loading and training model...")
    
    best_model_name, best_accuracy = load_and_train_model()
    
    print(f"✅ App ready! Best model: {best_model_name}")
    print("🌐 Starting Flask server...")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)