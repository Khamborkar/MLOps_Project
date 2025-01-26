import subprocess
import time
import os

# Path to model.py and app.py
model_script = 'src/model.py'
app_script = 'app.py'

# Step 1: Run model.py to train the model and save it
print("Training model...")

# Run model.py and wait for it to complete
process = subprocess.Popen(['python', model_script])
process.communicate()  # Wait for model.py to finish

# Ensure model files are saved before starting Flask app
if os.path.exists('model.h5') and os.path.exists('tokenizer.pkl'):
    print("Model and tokenizer saved successfully.")
else:
    print("Model or tokenizer files are missing. Exiting...")
    exit(1)

# Step 2: Run app.py (Flask app)
print("Starting Flask app...")
subprocess.run(['python', app_script])
