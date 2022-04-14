
The entire process is described in the following Diagram.



![](/images/scheme.png)

## Blood vessels segmentation
Blood vessels and segmented in order to obtain binary masks. Code implementation can be found in (**segmentation.py**) [see segmentation module documentation and examples](Modules/segmentation.md).

## Tumors segmentation
Tumor are segmented and may be post-processed (e.g. splitted to core and periphery brain regions). Code implementation can be found in (**segmentation.py**) [see segmentation module documentation and examples](Modules/segmentation.md).

## Distance Map
Once the blood vessels are segmented, an outer distance transform is performed to calculate the distance from blood vessel. This step is implemented in (**distance_transform.py**) script. [Documentation and examples of distance transfrom module](Modules/distance_transform.md).

## Data Collection And Aggregation
Once the distance transform of blood vessels is calculated and tumor are segmented into masks. The final profile is calculated. [see profiles module documentation and examples](Modules/profiles.md).