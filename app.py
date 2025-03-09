from flask import *
import os
from datetime import datetime

app = Flask(__name__)

# Ensure uploads directory exists
UPLOADS_DIR = 'uploads'
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if request.method == 'POST':
        t1_file_name = request.form.get('t1_file')
        t2_file_name = request.form.get('t2_file')
        t1c_file_name = request.form.get('t1c_file')
        flair_file_name = request.form.get('flair_file')

        if not all([t1_file_name, t2_file_name, t1c_file_name, flair_file_name]):
            return render_template('error.html', error_message="Please assign all four MRI images."), 400

        uploaded_files = request.files.getlist('scans')
        file_dict = {file.filename: file for file in uploaded_files}

        try:
            t1_file = file_dict[t1_file_name]
            t2_file = file_dict[t2_file_name]
            t1c_file = file_dict[t1c_file_name]
            flair_file = file_dict[flair_file_name]

            valid_extensions = {'.nii', '.nii.gz'}
            for file in [t1_file, t2_file, t1c_file, flair_file]:
                if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
                    return render_template('error.html', 
                                        error_message=f"Invalid file type: {file.filename}. Please upload .nii or .nii.gz files."), 400
        except KeyError:
            return render_template('error.html', error_message="Invalid file assignments."), 400

        # Create timestamp folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        upload_folder = os.path.join(UPLOADS_DIR, timestamp)
        os.makedirs(upload_folder, exist_ok=True)

        # Save files with standardized names
        t1_file.save(os.path.join(upload_folder, 'T1.nii.gz'))
        t2_file.save(os.path.join(upload_folder, 'T2.nii.gz'))
        t1c_file.save(os.path.join(upload_folder, 'T1C.nii.gz'))
        flair_file.save(os.path.join(upload_folder, 'FLAIR.nii.gz'))

        # Redirect to confirm upload page with timestamp
        return redirect(url_for('confirm_upload', folder=timestamp))

    return render_template('diagnose.html')

@app.route('/diagnose_segmented', methods=['POST'])
def diagnose_segmented():
    if request.method == 'POST':
        t1_file_name = request.form.get('t1_file')
        seg_file_name = request.form.get('seg_file')

        if not all([t1_file_name, seg_file_name]):
            return render_template('error.html', error_message="Please assign both T1 and Segmentation images."), 400

        uploaded_files = request.files.getlist('scans')
        file_dict = {file.filename: file for file in uploaded_files}

        try:
            t1_file = file_dict[t1_file_name]
            seg_file = file_dict[seg_file_name]

            valid_extensions = {'.nii', '.nii.gz'}
            for file in [t1_file, seg_file]:
                if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
                    return render_template('error.html', 
                                        error_message=f"Invalid file type: {file.filename}. Please upload .nii or .nii.gz files."), 400
        except KeyError:
            return render_template('error.html', error_message="Invalid file assignments."), 400

        # Create timestamp folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        upload_folder = os.path.join(UPLOADS_DIR, timestamp)
        os.makedirs(upload_folder, exist_ok=True)

        # Save files with standardized names
        t1_file.save(os.path.join(upload_folder, 'T1.nii.gz'))
        seg_file.save(os.path.join(upload_folder, 'Segmentation.nii.gz'))

        # Redirect to confirm upload page with timestamp
        return redirect(url_for('confirm_upload', folder=timestamp))

    return redirect(url_for('diagnose'))

@app.route('/confirm_upload/<folder>')
def confirm_upload(folder):
    return render_template('confirm_upload.html', folder=folder)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('about.html')

@app.route('/results')
def results():
    return render_template('result.html')