# Seizure Detection

Create a folder named "seizure-data". Unzip the data and place the unzipped data files in the "seizure-data" folder, it should look like this

```
seizure-detection/
train.py
seizure_detection.py
clips.tar.gz
seizure-data/
Dog_1/
Dog_1_ictal_segment_1.mat
Dog_1_ictal_segment_2.mat
Dog_1_interictal_segment_1.mat
Dog_1_interictal_segment_2.mat
Dog_1_test_segment_1.mat
Dog_1_test_segment_2.mat
Dog_2/
...
```

Try to run the code on the python version 2.7.x because the hickle doesnt work properly on other version and hickle is needed to cache the files.

Process to run it on a particular version of python 2.7.x is as :
Step 1: Download and Install Python 2.7
Download Python 2.7 from the official archive:

```
Python 2.7.18 Windows Installer --> https://www.python.org/ftp/python/2.7.18/python-2.7.18.msi
```

Run the Installer:

Check the option "Add Python to PATH" during installation.

If you forgot to add it to PATH, you can manually add C:\Python27 and C:\Python27\Scripts to your system PATH.

Verify Installation: Open Command Prompt (cmd) and run:

```
python --version
It should display:
Python 2.7.18
```

Step 2: Install virtualenv

Open Command Prompt and install virtualenv using pip:

```
pip install virtualenv
```

If pip is not found, try:

```
python -m ensurepip
python -m pip install --upgrade pip
python -m pip install virtualenv
```

Verify Installation:

```
virtualenv --version
```

Step 3: Create a Virtual Environment

```
Navigate to the desired project directory:
cd C:\path\to\your\project
```

Create a virtual environment:

```
virtualenv venv
```

This will create a folder venv containing a separate Python environment.

Step 4: Activate the Virtual Environment
On Windows (Command Prompt):

```
venv\Scripts\activate
```

On Windows (PowerShell):

```
powershell
venv\Scripts\Activate.ps1
```

```
(If you get a security error, run Set-ExecutionPolicy Unrestricted -Scope Process in PowerShell.)
```

Step 5: Verify Virtual Environment
Check if Python is using the virtual environment:

```
python --version
```

It should still display Python 2.7.18, but now it's inside the virtual environment.

Install packages inside the virtual environment:

```
pip install requests
```

Ensure the virtual environment is activated, then run:

```
python -m train
```

💡 This will:

Train a separate classifier for each patient.

Save them in the data-cache/ directory.
Verify training output:
dir data-cache

To exit the virtual environment, run:

```
deactivate
```

### Required

- python 2.7+ with virtualenv
- hickle==2.1.0
- numpy==1.14.0
- scikit-learn==0.19.1
- scipy==1.0.0

### Optional (to try out various data transforms)

- pywt (for Daubechies wavelet)
- scikits talkbox (for MFCC)

### Installing dependencies

Setup the virtualenv and install the dependencies

```
virtualenv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model and make predictions

Activate the virtualenv

```
. venv/bin/activate
```

```
seizure-data/
  Dog_1/
    Dog_1_ictal_segment_1.mat
    Dog_1_ictal_segment_2.mat
    ...
    Dog_1_interictal_segment_1.mat
    Dog_1_interictal_segment_2.mat
    ...
    Dog_1_test_segment_1.mat
    Dog_1_test_segment_2.mat
    ...

  Dog_2/
  ...
```

The directory name of the data should match the value in SETTINGS.json under the key `data-dir`.

Then simply run:

```
python -m train
```

One classifier is trained for each patient, and dumped to the data-cache directory.

```
data-cache/classifier_Dog_1_fft-with-time-freq-corr-1-48-r400-usf-gen1_rf3000mss1Bfrs0.pickle
data-cache/classifier_Dog_2_fft-with-time-freq-corr-1-48-r400-usf-gen1_rf3000mss1Bfrs0.pickle
...
data-cache/classifier_Patient_8_fft-with-time-freq-corr-1-48-r400-usf-gen1_rf3000mss1Bfrs0.pickle
```

Although using these classifiers outside the scope of this project is not very straightforward.

More convenient is to run the predict script.

```
python -m predict
```

This will take at least 2 hours.

Predictions are made using the test segments found in the data directory. They
are iterated over starting from 1 counting upwards until no file is found.

i.e.

```
seizure-data/
  Dog_1/
    Dog_1_test_segment_1.mat
    Dog_1_test_segment_2.mat
    ...
    Dog_1_test_segment_3181.mat
```

To make predictions on a new dataset, simply replace these test segments with new ones.
The files must numbered sequentially starting from 1 otherwise it will not find all of
the files.

This project uses a custom task system which caches task results to disk using hickle format and
falling back to pickle. First a task's output will be checked if it is in the data cache on disk,
and if not the task will be executed and the data cached.

See `seizure/tasks.py` for the custom tasks defined for this project. More specifically the
`MakePredictionsTask` depends on `TrainClassifierTask`, which means `predict.py` will train
and dump the classifiers as well as make predictions.

## Run cross-validation

```
python -m cross_validation
```

Cross-validation set is obtained by splitting on entire seizures. For example if there are 4 seizures,
3 seizures are used for training and 1 is used for cross-validation.

## SETTINGS.json

```
{
  "data-dir": "seizure-data",
  "data-cache-dir": "data-cache",
  "submission-dir": "./submissions"
}
```

- `data-dir`: directory containing the downloaded data
- `data-cache-dir`: directory the task framework will store cached data
- `submission-dir`: directory submissions are written to
