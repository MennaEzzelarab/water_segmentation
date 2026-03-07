import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pyngrok import ngrok, conf
from inference import load_model, run_prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload size

ALLOWED_EXTENSIONS = {'tif', 'tiff'}

# Ensure necessary directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# Load the model once when the app starts
model_path = os.path.join('model', 'final_UNet_ResNet34.pth')
print(f"Loading model from {model_path}...")
model, device = load_model(model_path)
print("Model loaded successfully.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Generate visualization and save it
            output_filename = f"pred_{filename.rsplit('.', 1)[0]}.png"
            output_filepath = os.path.join(app.config['STATIC_FOLDER'], output_filename)
            
            # Predict, save visualization, and get actual surface area
            water_area_sqm = run_prediction(model, device, filepath, output_filepath)
            
            return render_template('index.html', uploaded_image=output_filename, water_area=water_area_sqm)
            
    return render_template('index.html', uploaded_image=None, water_area=None)

@app.route('/static/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    # If an ngrok authtoken is provided, start a public tunnel.
    # Otherwise, just run the Flask app locally.
    ngrok_token = os.environ.get("NGROK_AUTHTOKEN")
    if ngrok_token:
        conf.get_default().auth_token = ngrok_token

        # Kill any existing tunnels to avoid ERR_NGROK_334
        try:
            tunnels = ngrok.get_tunnels()
            for tunnel in tunnels:
                ngrok.disconnect(tunnel.public_url)
        except Exception:
            pass

        try:
            public_url = ngrok.connect(5000)
            print("=" * 50)
            print(f"🌍 Your public URL is: {public_url.public_url}")
            print("Share this link with anyone to let them test your model!")
            print("=" * 50)
        except Exception as e:
            print("⚠️  ngrok failed to start. Running locally without a public URL.")
            print(f"   {type(e).__name__}: {e}")
    else:
        print("⚠️  NGROK_AUTHTOKEN not set. Running locally on http://127.0.0.1:5000")

    # Run the app without the reloader because ngrok connection should only happen once
    app.run(host="0.0.0.0", port=5000, debug=False)
