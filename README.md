# Cascade RCNN

### This work builds on [tf-eager-fasterrcnn](https://github.com/Viredery/tf-eager-fasterrcnn.git)

Faster R-CNN R-101-FPN model was implemented with TensorFlow2.0 Eager Execution. 

Cascade RCNN model was implemented with TensorFlow2.0 Eager Execution. 

# Requirements

- Cuda 10.0
- Python 3.5
- TensorFlow 2.0.0
- cv2

# Usage

see `train_cascade_rcnn.ipynb, train_faster_rcnn.ipynb`, `inspect_model.ipynb` and `eval_model.ipynb`


### Download trained Faster R-CNN

- [百度网盘](https://pan.baidu.com/s/1I5PGkpvnDSduJnngoWuktQ)
- [Google Drive](https://drive.google.com/file/d/1yCF-BqqM2x3bqWlJmAyDM-HuhDcLzt0t/view?usp=sharing)

# ToDO

- [ ] Muti-Scaling Training
- [ ] Pseudo Labeling
- [x] GHM-C loss
- [x] GHM-R loss
- [ ] Statistic Analysis of Dataset
- [X] WBF
- [ ] TTA
- [X] CutOut
- [X] MixUp
- [ ] Dilated Conv
- [X] SAG

# Acknowledgement

This work builds on many excellent works, which include:

- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
