# MultiSig

A classification architecture for multi-modal data. Each data modality is tokenized via signature methods. A decoder then performs two-task classification: label and data type. The use of a shared encoder proves especially useful for low-data environments with unbalanced data modalities. Currently supports image (.jpg), video (.mp4) and audio (.wav) data types. The signature tokenizations are extensions of the ideas discussed in ImageSig (https://arxiv.org/abs/2205.06929).

![Alt text](./full_architecture.png?raw=true)

The architecture was tested on a (quite unbalanced) dataset with the following structure:
```{bash}
data
├── training_set
│   ├── bird (1000 .jpg / 15 .mp4 / 8 .wav)
│   ├── cat  (5000 .jpg / 65 .mp4 / 5 .wav)
│   └── dog  (5000 .jpg /  2 .mp4 / 4 .wav)
└── test_set
    ├── bird (1000 .jpg /  0 .mp4 / 3 .wav)
    ├── cat  (1000 .jpg /  0 .mp4 / 3 .wav)
    └── dog  (1000 .jpg /  0 .mp4 / 0 .wav)
```

This work was produced as part of a 2 week industry mini-project in collaboration with [DataSig](https://www.datasig.ac.uk/) and supervised by [Dr Mohamed Ibrahim](https://www.datasig.ac.uk/people/mohamed-ibrahim). [Presentation](https://drive.google.com/drive/folders/1AhpZyyTdUDGXCrdTMwecrSVEJSUtsjiX).
