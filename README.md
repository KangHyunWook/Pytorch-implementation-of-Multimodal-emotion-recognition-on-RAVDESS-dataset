# Pytorch-implementation-of-Multimodal-emotion-recognition-on-RAVDESS-dataset

Description: This code performs emotion recognition using two modalities, which are audio and vision extracted while a person is speaking.

<h2>How to run</h2>

```
python train.py --data_path [path_to au_mfcc.pkl file]
```

shared encoder + private encoder:
acc: 0.8556

shared encoder only:
acc: 0.8444
