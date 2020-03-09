# Image segmentation with ResUNet
## Introduction
Image **segmentation** is very important area of machine learning but
at the same time a very **challenging task**: 
On the one hand, **global features** have to be **processed** by the model so that the objects in the image are recognized correctly, and 
on the other hand the model has to **keep local information** so that the individual pixels of the image can be assigned to the correct class.
<br>
The U-net architecture is adjusted to fulfill both requirements and is therefor used in many cases for segmentation problems. 
But the final model performance is also strongly influenced by the respective backbone model (resp. the en- and decoder architecture). 
For ResUNet special building blocks are designed to aggregate information on different scales which shall 
 provide advantages for object detection and context learning. The model is introduced in the following [paper](https://arxiv.org/pdf/1904.00592.pdf).
 ## Model 
 ### ResUNet Block
 
![ResUnet](resunet.png)
<br>
In contrast to standard Resnet blocks, Resunet uses blocks with multiple (usually 4) parallel branches which are all merged
in the end by addition. The branches differ from each other by a varying dilation sizes d<sub>i</sub> used for the 
convolutions (usually d<sub>i</sub> in {1,3,15,31}) while kernel size and stride are kept constant. Using a larger dilation size leads
to a larger receptive field while keeping the necessary parameters low. The idea of using different sized dilation is driven from the 
idea to aggregate features of different scales at the same time while in case of a sequential processing with fixed kernel parameters (kernel size,
stride, dilation) several pooling operations would be necessary in between to cover a comparable receptive field.
 ### Encoder and Decoder structure
![ResUnet](ResImg.png)
<br>
The encoder of Resunet consist of a sequence of residual blocks (introduced in the previous section) followed by a downsampling layer
implemented by a 2-dim. Convolution with stride 2. 
<br>
The decoder, on the other hand, is a sequence of residual blocks followed by upsampling layers 
(consisting of a 2-dim Convolution (kernel size 1x1) for reducing the channel number by fact. 2 and a simple upsampling operation) and a process where
corresponding encoder and decoder outputs are combined. The combination process takes place by concatenating the outputs and by processing them by 
a 1x1 Convolution.
# Dataset
To train and test the model we use the image data of the [Severstal: Steel Defect Detection]('https://www.kaggle.com/c/severstal-steel-defect-detection/overview')
competition. The dataset consists of images showing steel sheet surfaces which partly have one or more of 4 different defect types. The aim is to mark the image
pixels related to defects with the correct defect label. Hence, the model output and targets have shape (batch_size,img_height,img_width,4) where the 4 channels
 correspond to the 4 error types: The k-th target channel (k in {0,1,2,3}) contains a binary masks which marks pixels related to defect of type k with value 1 while the remaining 
 pixels are filled with 0.
The pictures below show 2 sample steel sheets which are shown 4 times in a row with respectively 1 of the 4 related target masks placed over it. 
Thus, e.g. the first sheet has errors of type 3 (marked in green) and of type 4 (marked in purple) while the second one has only type 4 defects.
![steel](steel.png)
![steel1](steel1.png)