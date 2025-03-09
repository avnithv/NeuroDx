from flask import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/diagnose', methods=['GET', 'POST'])
def diagnose():
    if request.method == 'POST':
        # Get the assigned files from hidden inputs
        t1_file_name = request.form.get('t1_file')
        t2_file_name = request.form.get('t2_file')
        t1c_file_name = request.form.get('t1c_file')
        flair_file_name = request.form.get('flair_file')

        if not all([t1_file_name, t2_file_name, t1c_file_name, flair_file_name]):
            return render_template('error.html', error_message="Please assign all four MRI images."), 400

        # Get the original files from the input
        uploaded_files = request.files.getlist('scans')
        file_dict = {file.filename: file for file in uploaded_files}

        # Match filenames to files and validate extensions
        try:
            t1_file = file_dict[t1_file_name]
            t2_file = file_dict[t2_file_name]
            t1c_file = file_dict[t1c_file_name]
            flair_file = file_dict[flair_file_name]

            # Validate file extensions
            valid_extensions = {'.nii', '.nii.gz'}
            for file in [t1_file, t2_file, t1c_file, flair_file]:
                if not any(file.filename.lower().endswith(ext) for ext in valid_extensions):
                    return render_template('error.html', 
                                        error_message=f"Invalid file type: {file.filename}. \n Please upload .nii or .nii.gz files."), 400
        except KeyError:
            return render_template('error.html', error_message="Invalid file assignments."), 400
        
        # Process the files (placeholder)
        print(f"T1: {t1_file.filename}, T2: {t2_file.filename}, "
              f"T1C: {t1c_file.filename}, FLAIR: {flair_file.filename}")
        # Example: Save files for processing
        # os.makedirs('uploads', exist_ok=True)
        # t1_file.save(os.path.join('uploads', t1_file.filename))
        # t2_file.save(os.path.join('uploads', t2_file.filename))
        # t1c_file.save(os.path.join('uploads', t1c_file.filename))
        # flair_file.save(os.path.join('uploads', flair_file.filename))
        return redirect(url_for('diagnose'))  # Replace with results page

    return render_template('diagnose.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('about.html')

@app.route('/results')
def results():
    return render_template('result.html')