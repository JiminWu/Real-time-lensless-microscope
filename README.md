# Real-time-lensless-microscope
The codes for the real-time lensless microscope ([link](https://opg.optica.org/boe/abstract.cfm?doi=10.1364/BOE.490199)), proposed in:

Biomedical Optics Express: "Real-time, deep-learning aided lensless microscope", Jimin Wu, Vivek Boominathan, Ashok Veeraraghavan* and Jacob T. Robinson*

</sub> * Corresponding authors </sub>

### Requirements
python 3.8+
pytorch 1.7+

### Included Examples
We provide a pre-trained model with several exapmles of similated (.png) data with ground truth and real captured data (.mat) in /sample_data/, along with the corresponding point spread functions (.mat). Pre-trained model can be downloaded from Google Drive [link](https://drive.google.com/file/d/1G-uBnOX-nSZ1aTTI0V3h_RZmSD3sBM5y/view?usp=drive_link).

### Real-time GUI
Current GUI works with ImagingSource DMM 37UX178-ML board-level camera (see more at ImagingSource support [link](https://github.com/TheImagingSource/IC-Imaging-Control-Samples/tree/master/Python)). qt5-simple.py provides a live stream test of the camera. 

### Acknowledgements
This repository drew inspiration and help from following resources:
* FlatNet: https://siddiquesalman.github.io/flatnet/
* MultiWienerNet: https://waller-lab.github.io/MultiWienerNet/
* Unet: https://github.com/milesial/Pytorch-UNet
* ImagingSource sensor Pytorch support: https://github.com/TheImagingSource/IC-Imaging-Control-Samples/tree/master/Python

### Contact Us
In case of any queries regarding the code, please reach out to [Jimin](mailto:jimin.wu@rice.edu).
Other raw and analysed data are available for research purpose from corresponding author upon reasonable request.
