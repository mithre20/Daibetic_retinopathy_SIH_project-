# Technical Support and Troubleshooting

## Common Technical Issues and Solutions

### 1. Backend Server Disconnected / "Failed to Fetch"
- **Symptom**: Clicking Run Screening or asking a chat question displays a connection error.
- **Cause**: The Python Flask backend is not running on port 5000.
- **Solution**: Open your terminal, activate the Python virtual environment (`.\.venv\Scripts\Activate.ps1`), and launch the server using `python backend/app.py`. Ensure port 5000 is open and not blocked by local firewalls.

### 2. Grad-CAM or SHAP Generation Timeout
- **Symptom**: The explainability loading spinner runs for more than 15 seconds.
- **Cause**: Running complex SHAP sampling (e.g. 50-100 samples) on an entry-level CPU without AVX2 or CUDA acceleration.
- **Solution**: The system defaults to 30 samples for optimal speed. Allow up to 10 seconds for gradient calculation on CPU.

### 3. IQA Rejection Loop
- **Symptom**: Every uploaded image is marked as "Poor Quality" or "Unusable".
- **Cause**: Image resolution is too low, camera lens is smudged, or the image format is unsupported.
- **Solution**: Use direct digital fundus exports (at least 512x512 pixels, preferably 1500x1500+). Ensure the lens of the fundus camera is clean.

### 4. 3D Retinal Visualization Stuttering on Hero Page
- **Symptom**: Low frame rate on the 3D eye model in Station 01.
- **Cause**: Hardware acceleration disabled in the browser.
- **Solution**: Open browser settings, search for "Hardware Acceleration", and ensure "Use graphics acceleration when available" is turned ON.

### 5. Chatbot Widget Disconnected
- **Symptom**: Chat messages return "I'm having trouble connecting right now."
- **Cause**: Backend API endpoint `/api/chat` is unreachable.
- **Solution**: Check the backend terminal console for network exceptions or verify that `GET /api/chat/health` returns `{"status": "healthy"}`.
