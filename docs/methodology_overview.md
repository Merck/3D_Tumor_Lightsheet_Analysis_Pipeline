
## Tumor Lightsheet data analysis pipeline: 

**Goal: Analysis of of biologic penetration in intact/whole tumors in the context of vasculature using using 3D Histology (tissue clearing + lightsheet microscopy).**

To obtain estimate  biologic penetration in tumors in the context of vasculature we developed an image analysis pipeline broadly based on prior work by Dobosz et al.[^1] We share our modular python-based image analysis approach for other researchers. 

Our lightsheet image analysis pipeline consisted of four main steps – Blood vessels segmentation, Tumour boundary segmentation, Vascular distance map creation, and Data aggregation, respectively. In this section, we describe each step of the pipeline individually


### Diagram of Our Pipeline
<!-- ![image](/images/pipeline.png =100x20) -->
<!-- <img src="/images/pipeline.png" width="2000"> -->
<img src="/images/pipeline.png">

### Analysis Pipeline Steps:

### Blood vessel detection:
* Vasculature in cleared tumors was immunolabeled using an anti-CD31 primary antibody and a fluorophore tagged secondary antibody. Images acquired in the vascular fluorescence channel need to be segmented to identify blood vessels (foreground) from background. 
* Variability in vascular fluorescence intensities makes segmentation challenging. A variety of segmentation methods were tested including simple intensity based thresholding, random forest based machine learning, and deep learning. 
* Once images have been segmented to create a binary mask of blood vessels and subsequently segmented digital objects, various quantitative attributes of the vessels such as location, size, etc. can extracted. This is also a critical step to support creation of distance maps in subsequent steps. 

* Code implementation can be found in (**segmentation.py**) [see segmentation module documentation and examples](Modules/segmentation.md).


### Tumor segmentation:
* Tumor tissue (foreground) can be detected from background by labeling all cell nuclei with a nuclear stain/DNA intercalating dye. 
* Once images have been segmented to create a binary mask of tumor tissue and subsequently a digital object, quantitative attributes such as tumor size and boundary can be extracted. 
* Segmented tumors may be post-processed (e.g. split into tumor core and periphery regions).

* Code implementation can be found in (**segmentation.py**) [see segmentation module documentation and examples](Modules/segmentation.md).

### Distance map:
* A 3D distance map is created to calculate the penetration of the antibody from the tumor vessels into the surrounding tissue. In this case, the distance transform operator, assigns each blood vessel pixel/voxel of the binary image with the distance to the nearest blood vessel pixel/voxel. 

* Documentation and examples of distance transform module (**distance_transform.py**) script. [Documentation and examples of distance transfrom module](Modules/distance_transform.md).


### Collect and aggregate data:
* Once the distance transform of blood vessels is calculated and tumor are segmented into masks.
* A final profile of drug intensity vs distance from vessel wall can be calculated.
* [see profiles module documentation and examples](Modules/profiles.md).


#
#### Refs
[^1]: Dobosz, M., Ntziachristos, V., Scheuer, W. & Strobel, S. **Multispectral Fluorescence Ultramicroscopy: Three-Dimensional Visualization and Automatic Quantification of Tumor Morphology, Drug Penetration, and Antiangiogenic Treatment Response**. Neoplasia 16, 1-U24, doi:10.1593/neo.131848 (2014).*
