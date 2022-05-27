# MultiSig

A classification architecture for multi-modal data. Signature-based tokenization of different data modalities feeds into a single encoder (especially useful for low-data environments with unbalanced data modalities). A decoder then performs two-task classification: label and  Currently supports image (.jpg), video (.mp4) and audio (.wav) data types. The signature tokenizations are extensions the ideas discussed in ImageSig (https://arxiv.org/abs/2205.06929).

![Alt text](./full_architecture.png?raw=true)

The architecture was tested on a dataset with the following structure:
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

