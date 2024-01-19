# Action-units-extraction

Description: This code extracts action units, which are movements of facial muscles while a person is speaking. The input is mp4 video and run as follows:

```
python au_extract.py -vid-folder [root_video_folder_path]
```

```
./RAVDESS/
          Actor_01
          Actor_02
```

The dataset link: https://zenodo.org/record/1188976#.ZALWUkpBxyY

To extract action units, install openface: https://github.com/TadasBaltrusaitis/OpenFace

This code had been run on Ubuntu 20.04 LTS

With those features extracted you can do sentiment analysis as follows:

1. sentiment analysis by analyzing movements of facial muscles: https://www.youtube.com/watch?v=XqvgfdJNcRg
2. multimodal sentiment analysis: https://www.youtube.com/watch?v=MRyzIuIxKzc

Later, I am going to upload a video about speech emotion recognition, for prerequisites check the following video for basic understanding of signal processing.

Fourier transform and absolute value of complex numbers: https://www.youtube.com/watch?v=CFSSA_8K6TM
