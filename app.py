from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from datetime import datetime
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import sys

import os
import sys
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.layers import Layer
from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch
from monai.bundle import ConfigParser
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, CenterSpatialCropd, NormalizeIntensityd, 
    ToTensord, AsDiscreted
)

import tensorflow as tf


import os
import sys
import numpy as np
import nibabel as nib
import keras
from flask import Flask, render_template, request, redirect, url_for, jsonify
import torch
from monai.bundle import ConfigParser
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, CenterSpatialCropd, NormalizeIntensityd, 
    ToTensord, AsDiscreted
)

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

        seg_output_path = os.path.join(upload_folder, 'Segmentation.nii.gz')
        segment_brats(t1_path, t2_path, t1c_path, flair_path, seg_output_path, timestamp)

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
        
        # print(filename, data.shape, data[:, :, i].shape, data, end=" ")
        slice_img=data[:, :, i]
        if len(slice_img.shape) == 3:
            slice_img = slice_img[2]
        # slice_img = np.squeeze(data[:, :, i])  # Ensure 2D [H, W]
        slice_path = os.path.join(file_slice_folder, f"{orientation}_{i}.png")
        plt.imsave(slice_path, slice_img, cmap='gray')
        slice_paths.append(f"/static/slices/{folder}/{filename_no_ext}/{orientation}_{i}.png")

    return jsonify({"slices": slice_paths})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/results')
def results():
    return render_template('result.html')

def segment_brats(t1_path, t2_path, t1c_path, flair_path, output_path, folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the bundle config
    config_file = "monai_bundles/brats_mri_segmentation/configs/inference.json"
    config = ConfigParser()
    config.read_config(config_file)
    
    # Minimal preprocessing for speed
    preprocess = Compose([
        LoadImaged(keys=["image"]),
        CenterSpatialCropd(keys=["image"], roi_size=(128, 128, 96)),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"])
    ])
    
    # Light postprocessing
    postprocess = Compose([
        AsDiscreted(keys=["pred"], argmax=True)  # Convert logits to labels
    ])

    # Prepare input data
    data_dict = {"image": [t1_path, t2_path, t1c_path, flair_path]}
    dataset = CacheDataset(data=[data_dict], transform=preprocess, cache_rate=1.0)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load PyTorch model
    model = config.get_parsed_content("network").to(device)
    model.load_state_dict(torch.load("monai_bundles/brats_mri_segmentation/models/model.pt", map_location=device))
    model.eval()

    # Load affine from T1 file directly
    t1_nifti = nib.load(t1_path)
    affine = t1_nifti.affine

    # Inference with fast sliding window
    with torch.no_grad():
        for data in dataloader:
            inputs = data["image"].to(device)  # [1, 4, 128, 128, 96]
            pred = sliding_window_inference(
                inputs,
                roi_size=(128, 128, 96),
                sw_batch_size=4,
                predictor=model,
                overlap=0.25
            )
            # Apply light postprocessing
            data["pred"] = pred
            data = postprocess(data)
            # Convert to numpy and save (ensure 3D: [128, 128, 96])
            pred_np = data["pred"].cpu().numpy()[0].astype('uint8')  # [128, 128, 96]
            seg_nifti = nib.Nifti1Image(pred_np, affine)
            nib.save(seg_nifti, output_path)
    
    return output_path


# Cast layer definition (unchanged)
class Cast(Layer):
    def __init__(self, dtype='float32', **kwargs):
        super(Cast, self).__init__(**kwargs)
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    def call(self, inputs):
        return tf.cast(inputs, dtype=self._dtype)

    def get_config(self):
        config = super(Cast, self).get_config()
        config.update({'dtype': self._dtype})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load glioma models
grading_model = tf.keras.models.load_model(
    'static/models/MixedType1_3d_cnn_model.h5',
    custom_objects={'Cast': Cast}
)
typing_model = tf.keras.models.load_model(
    'static/models/TumorTyping_3d_cnn_model.h5',
    custom_objects={'Cast': Cast}
)

def process_for_results(t1_path, seg_path, output_path):
    t1_img = nib.load(t1_path)
    seg_img = nib.load(seg_path)
    t1_data = t1_img.get_fdata()  # e.g., [128, 128, 96]
    seg_data = seg_img.get_fdata()  # [128, 128, 96], uint8
    
    combined_data = t1_data * seg_data  # [128, 128, 96]
    combined_data = np.expand_dims(combined_data, axis=(0, -1))  # [1, 128, 128, 96, 1]
    combined_data = (combined_data - combined_data.min()) / (combined_data.max() - combined_data.min() + 1e-8)
    
    # Run inference
    grading_pred = grading_model.predict(combined_data)  # [1, num_grades]
    typing_pred = typing_model.predict(combined_data)    # [1, num_types]
    
    # Get labels and confidences
    grading_labels = ['Grade I', 'Grade II', 'Grade III', 'Grade IV']
    typing_labels = ['Oligodendroglioma', 'Astrocytoma', 'Glioblastoma']
    
    grading_idx = np.argmax(grading_pred[0])
    typing_idx = np.argmax(typing_pred[0])
    
    grading_result = grading_labels[grading_idx]
    typing_result = typing_labels[typing_idx]
    
    grading_confidence = float(grading_pred[0][grading_idx]) * 100  # Convert to percentage
    typing_confidence = float(typing_pred[0][typing_idx]) * 100    # Convert to percentage
    
    # Save combined image
    combined_nifti = nib.Nifti1Image(combined_data[0, :, :, :, 0], t1_img.affine)
    nib.save(combined_nifti, output_path)
    
    return grading_result, typing_result, grading_confidence, typing_confidence

@app.route('/results/<folder>', methods=['GET', 'POST'])
def results(folder):
    upload_folder = os.path.join(UPLOADS_DIR, folder)
    if not os.path.exists(upload_folder):
        return render_template('error.html', error_message="Upload folder not found."), 400
    
    t1_path = os.path.join(upload_folder, 'T1.nii.gz')
    seg_path = os.path.join(upload_folder, 'Segmentation.nii.gz')
    combined_path = os.path.join(upload_folder, 'Combined.nii.gz')
    
    if not os.path.exists(t1_path) or not os.path.exists(seg_path):
        return render_template('error.html', error_message="Required files missing."), 400
    
    # Get results with confidences
    grading_result, typing_result, grading_confidence, typing_confidence = process_for_results(t1_path, seg_path, combined_path)
    
    return render_template('results.html', folder=folder, grading=grading_result, typing=typing_result,
                           grading_confidence=grading_confidence, typing_confidence=typing_confidence)