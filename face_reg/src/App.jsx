import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Users, Save, Eye, AlertCircle, CheckCircle } from 'lucide-react';

const FaceRecognitionApp = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [registeredUsers, setRegisteredUsers] = useState([]);
  const [isRegistering, setIsRegistering] = useState(false);
  const [registrationName, setRegistrationName] = useState('');
  const [recognitionData, setRecognitionData] = useState(null);
  const [notification, setNotification] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const recognitionIntervalRef = useRef(null);

  // Mock API endpoints (replace with your actual backend URLs)
  const API_BASE = 'http://localhost:5000';
  
  const showNotification = useCallback((message, type = 'info') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 3000);
  }, []);

  const initializeCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsConnected(true);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      showNotification('Failed to access camera', 'error');
    }
  }, [showNotification]);

  const fetchRegisteredUsers = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE}/api/users`);
      if (response.ok) {
        const users = await response.json();
        setRegisteredUsers(users);
      } else {
        // If backend is not ready, show empty state
        setRegisteredUsers([]);
      }
    } catch (error) {
      console.error('Error fetching users:', error);
      // Set empty array on network error
      setRegisteredUsers([]);
    } finally {
      setIsLoading(false);
    }
  }, [API_BASE]);

  const captureFrame = useCallback(() => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');
      
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      
      return canvas.toDataURL('image/jpeg');
    }
    return null;
  }, []);

  const startRecognition = useCallback(async () => {
    try {
      const imageData = captureFrame();
      
      if (!imageData) return;

      const response = await fetch(`${API_BASE}/api/recognize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData
        })
      });

      if (response.ok) {
        const result = await response.json();
        setRecognitionData(result);
        
        // Only show notification for successful recognition with good confidence
        if (result.name !== 'Unknown' && result.name !== 'No Face' && result.confidence > 70) {
          // Don't spam notifications - only show occasionally
          if (Math.random() > 0.95) { // Show roughly every 20 detections
            showNotification(`Recognized: ${result.name} (${result.confidence}%)`, 'success');
          }
        }
      }
    } catch (error) {
      // Silently handle recognition errors to avoid spam
      console.error('Recognition error:', error);
    }
  }, [captureFrame, API_BASE, showNotification]);

  // Initialize camera when component mounts
  useEffect(() => {
    initializeCamera();
    fetchRegisteredUsers();
    
    return () => {
      // Cleanup camera when component unmounts
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (recognitionIntervalRef.current) {
        clearInterval(recognitionIntervalRef.current);
      }
    };
  }, [initializeCamera, fetchRegisteredUsers]);

  // Continuous recognition effect
  useEffect(() => {
    // Clear existing interval
    if (recognitionIntervalRef.current) {
      clearInterval(recognitionIntervalRef.current);
    }

    if (!isRegistering && isConnected) {
      recognitionIntervalRef.current = setInterval(startRecognition, 1000);
    }

    return () => {
      if (recognitionIntervalRef.current) {
        clearInterval(recognitionIntervalRef.current);
      }
    };
  }, [isRegistering, isConnected, startRecognition]);

  const handleRegister = async () => {
    if (!registrationName.trim()) {
      showNotification('Please enter a name', 'error');
      return;
    }

    try {
      setIsLoading(true);
      const imageData = captureFrame();
      
      if (!imageData) {
        showNotification('Failed to capture image', 'error');
        return;
      }

      const response = await fetch(`${API_BASE}/api/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: registrationName,
          image: imageData
        })
      });

      const result = await response.json();

      if (response.ok) {
        showNotification(`Successfully registered ${registrationName}! (Total samples: ${result.total_samples})`, 'success');
        setRegistrationName('');
        setIsRegistering(false);
        fetchRegisteredUsers();
      } else {
        showNotification(result.error || 'Registration failed', 'error');
      }
    } catch (error) {
      console.error('Registration error:', error);
      showNotification('Network error - check if backend is running', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Camera className="w-8 h-8 text-blue-600" />
              <h1 className="text-2xl font-bold text-gray-800">Face Recognition System</h1>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-600">
                {isConnected ? 'Camera Connected' : 'Camera Disconnected'}
              </span>
            </div>
          </div>
        </div>

        {/* Notification */}
        {notification && (
          <div className={`mb-4 p-4 rounded-lg flex items-center space-x-2 ${
            notification.type === 'success' ? 'bg-green-100 text-green-800' :
            notification.type === 'error' ? 'bg-red-100 text-red-800' :
            'bg-blue-100 text-blue-800'
          }`}>
            {notification.type === 'success' ? <CheckCircle className="w-5 h-5" /> :
             notification.type === 'error' ? <AlertCircle className="w-5 h-5" /> :
             <AlertCircle className="w-5 h-5" />}
            <span>{notification.message}</span>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Video Feed */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4">Live Video Feed</h2>
              
              <div className="relative bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-96 object-cover"
                />
                
                {/* Recognition overlay */}
                {recognitionData && recognitionData.name !== 'Unknown' && recognitionData.name !== 'No Face' && recognitionData.confidence > 60 && (
                  <div className="absolute top-4 left-4 bg-green-600 text-white px-3 py-1 rounded-lg">
                    {recognitionData.name} ({recognitionData.confidence}%)
                  </div>
                )}
                
                {/* Registration mode overlay */}
                {isRegistering && (
                  <div className="absolute top-4 right-4 bg-blue-600 text-white px-3 py-1 rounded-lg">
                    Registration Mode: {registrationName}
                  </div>
                )}
              </div>

              {/* Controls */}
              <div className="mt-4 space-y-4">
                {!isRegistering ? (
                  <button
                    onClick={() => setIsRegistering(true)}
                    disabled={!isConnected || isLoading}
                    className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-lg transition duration-200"
                  >
                    Register New Face
                  </button>
                ) : (
                  <div className="space-y-3">
                    <input
                      type="text"
                      placeholder="Enter person's name"
                      value={registrationName}
                      onChange={(e) => setRegistrationName(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <div className="flex space-x-2">
                      <button
                        onClick={handleRegister}
                        disabled={isLoading || !registrationName.trim()}
                        className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-lg transition duration-200"
                      >
                        {isLoading ? 'Registering...' : 'Capture & Register'}
                      </button>
                      <button
                        onClick={() => {
                          setIsRegistering(false);
                          setRegistrationName('');
                        }}
                        className="flex-1 bg-gray-600 hover:bg-gray-700 text-white font-medium py-2 px-4 rounded-lg transition duration-200"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Stats */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold mb-3 flex items-center">
                <Users className="w-5 h-5 mr-2" />
                System Stats
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Registered Users:</span>
                  <span className="font-medium">{registeredUsers.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Status:</span>
                  <span className={`font-medium ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
                    {isConnected ? 'Active' : 'Inactive'}
                  </span>
                </div>
              </div>
            </div>

            {/* Current Recognition */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold mb-3 flex items-center">
                <Eye className="w-5 h-5 mr-2" />
                Current Recognition
              </h3>
              {recognitionData ? (
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Name:</span>
                    <span className="font-medium">{recognitionData.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Confidence:</span>
                    <span className="font-medium">{recognitionData.confidence}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Time:</span>
                    <span className="font-medium">{new Date().toLocaleTimeString()}</span>
                  </div>
                </div>
              ) : (
                <p className="text-gray-500">No face detected</p>
              )}
            </div>

            {/* Registered Users */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold mb-3">Registered Users</h3>
              {isLoading ? (
                <p className="text-gray-500">Loading...</p>
              ) : registeredUsers.length > 0 ? (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {registeredUsers.map((user, index) => (
                    <div key={user.face_id || index} className="p-2 bg-gray-50 rounded">
                      <div className="font-medium text-sm">{user.name}</div>
                      <div className="text-xs text-gray-500">
                        Time: {user.registration_time || 'N/A'}&nbsp;
                        Date: {user.registration_date || 'N/A'}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500">No users registered</p>
              )}
            </div>
          </div>
        </div>

        {/* Hidden canvas for image capture */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
    </div>
  );
};

export default FaceRecognitionApp;