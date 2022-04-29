
## Tumor Lightsheet data analysis pipeline: 

To obtain estimate  biologic penetration in tumors in the context of vasculature we developed an image analysis pipeline broadly based on prior work by Dobosz et al. [1] We share our modular python-based image analysis approach for other researchers. 

Our lightsheet image analysis pipeline consisted of four main steps – Blood vessels segmentation, Tumour boundary segmentation, Vascular distance map creation, and Data aggregation, respectively. In this section, we describe each step of the pipeline individually


### Diagram of Our Pipeline
![](docs/images/pipeline.png/pipeline.png)




## Individual Steps

### Blood vessels segmentation
Blood vessels and segmented in order to obtain binary masks. Code implementation can be found in (**segmentation.py**) [see segmentation module documentation and examples](Modules/segmentation.md).

### Tumors segmentation
Tumor are segmented and may be post-processed (e.g. splitted to core and periphery brain regions). Code implementation can be found in (**segmentation.py**) [see segmentation module documentation and examples](Modules/segmentation.md).

### Distance Map
Once the blood vessels are segmented, an outer distance transform is performed to calculate the distance from blood vessel. This step is implemented in (**distance_transform.py**) script. [Documentation and examples of distance transfrom module](Modules/distance_transform.md).

### Data Collection And Aggregation
Once the distance transform of blood vessels is calculated and tumor are segmented into masks. The final profile is calculated. [see profiles module documentation and examples](Modules/profiles.md).

#
#### Refs
[1]: Dobosz, M., Ntziachristos, V., Scheuer, W. & Strobel, S. Multispectral Fluorescence Ultramicroscopy: Three-Dimensional Visualization and Automatic Quantification of Tumor Morphology, Drug Penetration, and Antiangiogenic Treatment Response. Neoplasia 16, 1-U24, doi:10.1593/neo.131848 (2014).
