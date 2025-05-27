import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import json

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize face detection and recognition
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Storage for faces and names
        self.known_faces = []
        self.known_names = []
        self.face_id_counter = 0
        
        # Files to store data
        self.data_dir = "face_data"
        self.model_file = os.path.join(self.data_dir, "face_model.yml")
        self.names_file = os.path.join(self.data_dir, "names.json")
        self.log_file = os.path.join(self.data_dir, "recognition_log.json")
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # Load existing data
        self.load_data()
        
        print("Face Recognition System Initialized!")
        print("Commands:")
        print("'r' - Register new face")
        print("'s' - Save current model")
        print("'v' - View registered users")
        print("'q' - Quit")
    
    def load_data(self):
        """Load existing face data and model"""
        try:
            # Load names and registration data
            if os.path.exists(self.names_file):
                with open(self.names_file, 'r') as f:
                    names_data = json.load(f)
                    # Handle both old format (just names) and new format (with timestamps)
                    if isinstance(names_data, dict) and 'registrations' in names_data:
                        # New format with registration timestamps
                        self.known_names = [reg['name'] for reg in names_data['registrations']]
                        self.face_id_counter = names_data.get('counter', 0)
                        print(f"Loaded {len(self.known_names)} known faces with registration data")
                    else:
                        # Old format - just names list
                        self.known_names = names_data.get('names', []) if isinstance(names_data, dict) else names_data
                        self.face_id_counter = len(self.known_names)
                        print(f"Loaded {len(self.known_names)} known faces (legacy format)")
            
            # Load trained model
            if os.path.exists(self.model_file):
                self.face_recognizer.read(self.model_file)
                print("Loaded trained face recognition model")
        
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def save_data(self):
        """Save face data and model with registration timestamps"""
        try:
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
                # Check if we already have registration data for this person
                existing_reg = next((reg for reg in existing_registrations if reg.get('face_id') == i), None)
                
                if existing_reg:
                    # Use existing registration data
                    registrations.append(existing_reg)
                else:
                    # Create new registration entry (for newly added names)
                    registration = {
                        'face_id': i,
                        'name': name,
                        'registration_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'registration_date': datetime.now().strftime('%Y-%m-%d'),
                        'registration_time': datetime.now().strftime('%H:%M:%S')
                    }
                    registrations.append(registration)
            
            # Save in new format
            names_data = {
                'registrations': registrations,
                'counter': self.face_id_counter,
                'total_registered': len(self.known_names),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(self.names_file, 'w') as f:
                json.dump(names_data, f, indent=2)
            
            # Save model if we have training data
            if len(self.known_faces) > 0:
                self.face_recognizer.write(self.model_file)
            
            print("Data saved successfully with registration timestamps!")
        
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
            
            # Load existing log
            log_data = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            
            # Add new entry
            log_data.append(log_entry)
            
            # Keep only last 1000 entries
            if len(log_data) > 1000:
                log_data = log_data[-1000:]
            
            # Save log
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        
        except Exception as e:
            print(f"Error logging recognition: {e}")
    
    def register_face(self, face_img, name):
        """Register a new face with a name and timestamp"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                print("No face detected for registration!")
                return False
            
            # Use the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize face for consistency
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Add to training data
            self.known_faces.append(face_roi)
            self.known_names.append(name)
            
            # Create labels for training
            labels = list(range(len(self.known_names)))
            
            # Train the recognizer
            self.face_recognizer.train(self.known_faces, np.array(labels))
            
            # Get registration timestamp
            registration_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Log registration with detailed timestamp info
            print(f"Face registered for '{name}' at {registration_timestamp}")
            
            # Update face ID counter
            self.face_id_counter = len(self.known_names)
            
            return True
        
        except Exception as e:
            print(f"Error registering face: {e}")
            return False
    
    def recognize_face(self, face_roi):
        """Recognize a face and return name with confidence"""
        try:
            # Resize for consistency
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Predict
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
    def display_registered_users(self):
        """Display all registered users with their registration timestamps"""
        try:
            if os.path.exists(self.names_file):
                with open(self.names_file, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, dict) and 'registrations' in data:
                    print("\n=== REGISTERED USERS ===")
                    print(f"Total Users: {len(data['registrations'])}")
                    print("-" * 50)
                    
                    for reg in data['registrations']:
                        print(f"ID: {reg['face_id']}")
                        print(f"Name: {reg['name']}")
                        print(f"Registered: {reg['registration_timestamp']}")
                        print("-" * 30)
                    
                    print(f"Last Updated: {data.get('last_updated', 'N/A')}")
                else:
                    print("No registration data found or using legacy format")
            else:
                print("No registration file found")
        
        except Exception as e:
            print(f"Error displaying registered users: {e}")
    
    def get_registration_info(self, name):
        """Get registration information for a specific user"""
        try:
            if os.path.exists(self.names_file):
                with open(self.names_file, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, dict) and 'registrations' in data:
                    for reg in data['registrations']:
                        if reg['name'].lower() == name.lower():
                            return reg
            return None
        
        except Exception as e:
            print(f"Error getting registration info: {e}")
            return None
        
    def run(self):
        """Main loop for face recognition system"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Cannot access webcam!")
            return
        
        print("Face Recognition System Running...")
        print("Press 'r' to register a new face, 'v' to view users, 's' to save, 'q' to quit")
        
        registration_mode = False
        registration_name = ""
        last_recognition_time = {}
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            
            current_time = datetime.now()
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                if registration_mode:
                    # Registration mode - draw blue rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, f"Registering: {registration_name}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    # Recognition mode
                    if len(self.known_names) > 0:
                        name, confidence = self.recognize_face(face_roi)
                        
                        # Choose color based on recognition
                        if name != "Unknown" and confidence > 70:
                            color = (0, 255, 0)  # Green for known faces
                            
                            # Log recognition (but not too frequently for same person)
                            current_timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
                            if (name not in last_recognition_time or 
                                (current_time - last_recognition_time[name]).seconds > 30):
                                self.log_recognition(name, confidence, current_timestamp)
                                last_recognition_time[name] = current_time
                                print(f"Recognized: {name} (confidence: {confidence:.1f}%) at {current_timestamp}")
                        else:
                            color = (0, 0, 255)  # Red for unknown faces
                        
                        # Draw rectangle around face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Display name and confidence at top-left corner of bounding box
                        label = f"{name} ({confidence:.1f}%)"
                        cv2.putText(frame, label, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Display timestamp at bottom-right corner of bounding box
                        timestamp = current_time.strftime('%H:%M:%S')
                        cv2.putText(frame, timestamp, (x+w-80, y+h+15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    else:
                        # No trained faces yet
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 128, 128), 2)
                        cv2.putText(frame, "No trained faces", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # Display instructions
            instructions = [
                "Commands: 'r'-Register, 'v'-View Users, 's'-Save, 'q'-Quit",
                f"Known faces: {len(self.known_names)}",
                f"Registration mode: {'ON' if registration_mode else 'OFF'}"
            ]
            
            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Face Recognition System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Enter registration mode
                name = input("\nEnter person's name for registration: ").strip()
                if name:
                    registration_mode = True
                    registration_name = name
                    print(f"Registration mode ON for '{name}'. Position face in camera and press 'c' to capture.")
                else:
                    print("Invalid name entered.")
            elif key == ord('c') and registration_mode:
                # Capture and register face
                if len(faces) > 0:
                    success = self.register_face(frame, registration_name)
                    if success:
                        print(f"Successfully registered '{registration_name}'!")
                        self.save_data()  # Auto-save after registration
                    registration_mode = False
                    registration_name = ""
                else:
                    print("No face detected. Try again.")
            elif key == ord('s'):
                # Save data manually
                self.save_data()
            elif key == ord('v'):
                # View registered users
                print("\n" + "="*50)
                self.display_registered_users()
                print("="*50 + "\n")
            elif key == 27:  # ESC key
                registration_mode = False
                registration_name = ""
                print("Registration cancelled.")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Final save
        self.save_data()
        print("Face recognition system stopped.")

def main():
    try:
        system = FaceRecognitionSystem()
        system.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have opencv-contrib-python installed:")
        print("pip install opencv-contrib-python")

if __name__ == "__main__":
    main()