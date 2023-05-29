# Pytorch-implementation-of-Multimodal-emotion-recognition-on-RAVDESS-dataset

This work is in progress, the code may be updated. However, it provides technical details of how to fuse two modalities.

Description: This code performs emotion recognition using two modalities, which are audio and vision extracted while a person is speaking.

Youtube link: https://www.youtube.com/watch?v=MRyzIuIxKzc&lc=UgxOCzs7FV74n8LYnBt4AaABAg

<h2>How to run</h2>

```
python train.py --data_path [path_to au_mfcc.pkl file]
```

Above au_mfcc file contains mfcc features and the corresponding facial action units.
The action units are extracted using OpenFace
The code for preprocessing the raw mp4 data is to be uploaded.

<h2>Experimental results</h2>

shared encoder + private encoder:
acc: 0.8556

shared encoder only:
acc: 0.8444

audio only:
acc: 0.3167
