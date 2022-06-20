# Unsupervised Multi-View Gaze Representation Learning

This is the official project page of our CVPR Gaze 2022 workshop paper, winner of the best poster competition.

### Abstract

We present a method for unsupervised gaze representation learning from multiple synchronized views of a person's face. The key assumption is that images of the same eye captured from different viewpoints differ in certain respects while remaining similar in others. Specifically, the absolute gaze and absolute head pose of the same subject should be different from different viewpoints, while appearance characteristics and gaze angle relative to the head coordinate frame should remain constant. To leverage this, we adopt a cross-encoder learning framework, in which our encoding space consists of head pose, relative eye gaze, eye appearance and other common features. Image pairs which are assumed to have matching subsets of features should be able to swap those subsets among themselves without any loss of information, computed by decoding the mixed features back into images and measuring reconstruction loss. We show that by applying these assumptions to an unlabelled multi-view video dataset, we can generate more powerful representations than a standard gaze cross-encoder for few-shot gaze estimation. Furthermore, we introduce a new feature-mixing method which results in higher performance, faster training, improved testing flexibility with multiple views, and added interpretability with learned confidence.

### CVPR Gaze 2022 workshop materials

- [x] [Paper](docs/paper.pdf)
- [x] [Poster](docs/poster.pdf) (Winner - Best Poster)
- [x] [Slides](docs/slides.pdf)
- [x] [Video](https://youtu.be/gkYyOyiAB6k) (7 min summary)
- [x] Code - see [cvpr/README.md](cvpr/README.md)

### Citation

```
@InProceedings{Gideon_2022_CVPR_Gaze,
    author       = {Gideon, John and Su, Shan and Stent, Simon},
    title        = {Unsupervised Multi-View Gaze Representation Learning},
    booktitle    = {Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
    booksubtitle = {Workshop on Gaze Estimation and Prediction in the Wild},
    month        = {June},
    year         = {2022},
}
```

## License
This project is licensed under the Creative Commons Non Commercial License. See [LICENSE](LICENSE) file for further information.
