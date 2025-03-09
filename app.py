from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from datetime import datetime
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys

app = Flask(__name__)

UPLOADS_DIR = 'uploads'
SLICE_DIR = 'static/slices'
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SLICE_DIR, exist_ok=True)

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

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        upload_folder = os.path.join(UPLOADS_DIR, timestamp)
        os.makedirs(upload_folder, exist_ok=True)

        t1_path = os.path.join(upload_folder, 'T1.nii.gz')
        t2_path = os.path.join(upload_folder, 'T2.nii.gz')
        t1c_path = os.path.join(upload_folder, 'T1C.nii.gz')
        flair_path = os.path.join(upload_folder, 'FLAIR.nii.gz')

        t1_file.save(t1_path)
        t2_file.save(t2_path)
        t1c_file.save(t1c_path)
        flair_file.save(flair_path)

        return redirect(url_for('confirm_upload', folder=timestamp, is_segmented=0))
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

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        upload_folder = os.path.join(UPLOADS_DIR, timestamp)
        os.makedirs(upload_folder, exist_ok=True)

        t1_path = os.path.join(upload_folder, 'T1.nii.gz')
        seg_path = os.path.join(upload_folder, 'Segmentation.nii.gz')

        t1_file.save(t1_path)
        seg_file.save(seg_path)

        return redirect(url_for('confirm_upload', folder=timestamp, is_segmented=1))
    return redirect(url_for('diagnose'))

@app.route('/confirm_upload/<folder>')
def confirm_upload(folder):
    is_segmented = int(request.args.get('is_segmented', 0))
    upload_folder = os.path.join(UPLOADS_DIR, folder)
    print(folder, upload_folder, is_segmented)
    sys.stdout.flush()
    if not os.path.exists(upload_folder):
        return render_template('error.html', error_message="Upload folder not found."), 400

    return render_template('confirm_upload.html', folder=folder, is_segmented=is_segmented)

@app.route('/get_slices/<folder>/<filename>/<orientation>')
def get_slices(folder, filename, orientation):
    nifti_path = os.path.join(UPLOADS_DIR, folder, filename)
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Transpose data based on orientation
    if orientation == 'coronal':
        data = np.transpose(data, (1, 0, 2))  # Swap x and y
    elif orientation == 'sagittal':
        data = np.transpose(data, (2, 0, 1))  # Swap x and z
    # Axial is default (no transpose)

    slice_paths = []
    slice_folder = os.path.join(SLICE_DIR, folder)
    os.makedirs(slice_folder, exist_ok=True)
    
    filename_no_ext = filename.split('.')[0]  # e.g., 'T1' from 'T1.nii.gz'
    file_slice_folder = os.path.join(slice_folder, filename_no_ext)
    os.makedirs(file_slice_folder, exist_ok=True)
    
    for i in range(data.shape[2]):
        slice_img = data[:, :, i]
        slice_path = os.path.join(file_slice_folder, f"{orientation}_{i}.png")
        plt.imsave(slice_path, slice_img, cmap='gray')
        slice_paths.append(f"/static/slices/{folder}/{filename_no_ext}/{orientation}_{i}.png")

    return jsonify({"slices": slice_paths})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('about.html')

@app.route('/results')
def results():
    return render_template('result.html')