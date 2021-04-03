# Description
This Python Script tries to access your webcam and will tell for each detected face wheter it is smiling or not.
The Faces are recognised with a Haarcascade from OpenCV and the Smiles are detected with a self-trained CNN. The CNN is built upon the VGG16 and uses Transfer Learning to adjust itself to recognize Smiles.
The dataset is from [https://github.com/hromi/SMILEsmileD](https://github.com/hromi/SMILEsmileD).


# How to start (Windows)
```sh
$ ./venv/Scripts/Activate.ps1
(venv) $ pip install -r requirements.txt
(venv) $ python main.py
```

# How to reset model
Just delete the ./model/saved_model folder. The code will then retrain the model.
