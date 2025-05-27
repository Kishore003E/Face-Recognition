# api_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize face detection and recognition
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Storage for faces and names
        self.known_faces = []
        self.known_names = []
        self.face_labels = []  # Add labels list
        self.face_id_counter = 0
        self.is_trained = False  # Track if model is trained
        
        # Files to store data
        self.data_dir = "face_data"
        self.model_file = os.path.join(self.data_dir, "face_model.yml")
        self.names_file = os.path.join(self.data_dir, "names.json")
        self.log_file = os.path.join(self.data_dir, "recognition_log.json")
        self.faces_file = os.path.join(self.data_dir, "face_samples.pkl")
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Load existing data
        self.load_data()
        
        print("Face Recognition System Initialized!")
    
    def load_data(self):
        """Load existing face data and model"""
        try:
            # Load names and registration data
            if os.path.exists(self.names_file):
                with open(self.names_file, 'r') as f:
                    names_data = json.load(f)
                    if isinstance(names_data, dict) and 'registrations' in names_data:
                        self.known_names = [reg['name'] for reg in names_data['registrations']]
                        self.face_id_counter = names_data.get('counter', 0)
                        print(f"Loaded {len(self.known_names)} known faces with registration data")
                    else:
                        self.known_names = names_data.get('names', []) if isinstance(names_data, dict) else names_data
                        self.face_id_counter = len(self.known_names)
                        print(f"Loaded {len(self.known_names)} known faces (legacy format)")
            
            # Load face samples if they exist
            if os.path.exists(self.faces_file):
                import pickle
                with open(self.faces_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_faces = data.get('faces', [])
                    self.face_labels = data.get('labels', [])
                    print(f"Loaded {len(self.known_faces)} face samples")
            
            # Load trained model if it exists and we have faces
            if os.path.exists(self.model_file) and len(self.known_faces) > 0:
                self.face_recognizer.read(self.model_file)
                self.is_trained = True
                print("Loaded trained face recognition model")
        
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def save_data(self):
        """Save face data and model with registration timestamps"""
        try:
            # Save face samples
            import pickle
            face_data = {
                'faces': self.known_faces,
                'labels': self.face_labels
            }
            with open(self.faces_file, 'wb') as f:
                pickle.dump(face_data, f)
            
            # Load existing registration data to preserve timestamps
            existing_registrations = []
            if os.path.exists(self.names_file):
                with open(self.names_file, 'r') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, dict) and 'registrations' in existing_data:
                        existing_registrations = existing_data['registrations']
            
            # Create new format with registration timestamps
            registrations = []
            for i, name in enumerate(self.known_names):
                existing_reg = next((reg for reg in existing_registrations if reg.get('name') == name), None)
                
                if existing_reg:
                    registrations.append(existing_reg)
                else:
                    registration = {
                        'face_id': i,
                        'name': name,
                        'registration_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'registration_date': datetime.now().strftime('%Y-%m-%d'),
                        'registration_time': datetime.now().strftime('%H:%M:%S')
                    }
                    registrations.append(registration)
            
            names_data = {
                'registrations': registrations,
                'counter': self.face_id_counter,
                'total_registered': len(self.known_names),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(self.names_file, 'w') as f:
                json.dump(names_data, f, indent=2)
            
            # Save model if we have training data
            if self.is_trained and len(self.known_faces) > 0:
                self.face_recognizer.write(self.model_file)
            
            print("Data saved successfully!")
        
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def log_recognition(self, name, confidence, timestamp):
        """Log recognition events with timestamps"""
        try:
            log_entry = {
                'name': name,
                'confidence': float(confidence),
                'timestamp': timestamp,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M:%S')
            }
            
            log_data = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            
            log_data.append(log_entry)
            
            if len(log_data) > 1000:
                log_data = log_data[-1000:]
            
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        
        except Exception as e:
            print(f"Error logging recognition: {e}")
    
    def register_face(self, face_img, name):
        """Register a new face with a name and timestamp"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                print("No face detected for registration!")
                return False
            
            # Use the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Get the face ID for this person
            if name in self.known_names:
                face_id = self.known_names.index(name)
                print(f"Adding additional sample for existing user: {name}")
            else:
                face_id = len(self.known_names)
                self.known_names.append(name)
                print(f"Registering new user: {name} with ID: {face_id}")
            
            # Add face sample and corresponding label
            self.known_faces.append(face_roi)
            self.face_labels.append(face_id)
            
            # Train the recognizer with all samples
            if len(self.known_faces) > 0:
                print(f"Training with {len(self.known_faces)} samples and {len(self.face_labels)} labels")
                self.face_recognizer.train(self.known_faces, np.array(self.face_labels))
                self.is_trained = True
            
            self.face_id_counter = len(self.known_names)
            
            registration_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Face registered for '{name}' at {registration_timestamp}")
            
            return True
        
        except Exception as e:
            print(f"Error registering face: {e}")
            return False
    
    def recognize_face(self, face_roi):
        """Recognize a face and return name with confidence"""
        try:
            if not self.is_trained or len(self.known_faces) == 0:
                return "Unknown", 0
                
            face_roi = cv2.resize(face_roi, (100, 100))
            label, confidence = self.face_recognizer.predict(face_roi)
            
            # Lower confidence means better match (distance)
            if confidence < 100 and label < len(self.known_names):
                name = self.known_names[label]
                confidence_percent = max(0, 100 - confidence)
                return name, confidence_percent
            else:
                return "Unknown", 0
        
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return "Unknown", 0

# Initialize face recognition system
face_system = FaceRecognitionSystem()

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all registered users"""
    try:
        if os.path.exists(face_system.names_file):
            with open(face_system.names_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'registrations' in data:
                    return jsonify(data['registrations'])
        return jsonify([])
    except Exception as e:
        print(f"Error getting users: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def register_face():
    """Register a new face"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        image_data = data.get('image', '')
        
        if not name or not image_data:
            return jsonify({'error': 'Name and image are required'}), 400
        
        # Decode base64 image
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'error': 'Invalid image data'}), 400
                
        except Exception as e:
            print(f"Error decoding image: {e}")
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Register the face
        success = face_system.register_face(frame, name)
        
        if success:
            face_system.save_data()
            return jsonify({
                'message': f'Successfully registered {name}', 
                'success': True,
                'total_samples': len(face_system.known_faces),
                'users_count': len(face_system.known_names)
            })
        else:
            return jsonify({'error': 'Failed to register face - no face detected'}), 400
            
    except Exception as e:
        print(f"Error in register_face: {e}")
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Recognize a face in the image"""
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        
        if not image_data:
            return jsonify({'error': 'Image is required'}), 400
        
        # Decode base64 image
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'error': 'Invalid image data'}), 400
                
        except Exception as e:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_system.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        
        if len(faces) > 0:
            # Use the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            
            # Recognize the face
            name, confidence = face_system.recognize_face(face_roi)
            
            # Log recognition if it's a known face
            if name != "Unknown" and confidence > 60:  # Lower threshold for better detection
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                face_system.log_recognition(name, confidence, timestamp)
            
            return jsonify({
                'name': name,
                'confidence': round(confidence, 1),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'face_detected': True,
                'bounding_box': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                }
            })
        else:
            return jsonify({
                'name': 'No Face',
                'confidence': 0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'face_detected': False,
                'bounding_box': None
            })
            
    except Exception as e:
        print(f"Error in recognize_face: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recognition logs"""
    try:
        if os.path.exists(face_system.log_file):
            with open(face_system.log_file, 'r') as f:
                logs = json.load(f)
                return jsonify(logs[-50:])  # Return last 50 logs
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'total_users': len(face_system.known_names),
        'total_samples': len(face_system.known_faces),
        'is_trained': face_system.is_trained,
        'system_ready': len(face_system.known_faces) > 0
    })

if __name__ == '__main__':
    print("Starting Face Recognition API Server...")
    print("Available endpoints:")
    print("- GET  /api/users")
    print("- POST /api/register")
    print("- POST /api/recognize") 
    print("- GET  /api/logs")
    print("- GET  /api/status")
    app.run(debug=True, host='0.0.0.0', port=5000)