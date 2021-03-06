### Code use pre-requisites and Installation instructions

Prior to running the tumor lightsheet image analysis pipeline, the following pre-requisites must be met: 

* A) Installation of the python ***environment named 3d*** that contains all the necessary python packages and modules required to run this code.  
* B) Organization of lightsheet data in the recommended ***folder architecture***. 
* C) Creation of a ***configuration.json file***. 


Each of these pre-requisites are described in detail below.  

#### A) Installation of the required python environment:  

* Guide for the installation [can be found at this link](Installation Process.md)

#### B) Folder Structure

Due to the large size of the lightsheet datasets, the code is written to allow output from each step of the analysis pipeline to directly save the results to the hard disk. These 
results can then be immediately loaded into memory. The main reason for this approach is the ability to perform the analysis on a local, moderately powerful computer. 

This overcomes the limitations imposed by limited RAM memory. As a consequence, it is necessary to use a fixed folder structure for the raw data and configuration file provided as an input. The recommended folder structure is illustrated with an example below: 

3D_Tumor_Lightsheet_Analysis_Pipeline  (can be replaced with your root directory name) 
* When analyzing one particular study, the following **folders structure** of three channels (vessels, tumors, virus) including the **config.json** file is expected.

* Expected Folder Structure

    The **root directory** is considered the **3D_Tumor_Lightsheet_Analysis_Pipeline** folder (see tree diagram below)

    ```python
    3D_Tumor_Lightsheet_Analysis_Pipeline
    └─ data
       └─ your_study_name
            └─ config.json
            └─ source
                └─raw
                   └─tumor
                   │   └─ 5IT-4X_Ch2_z0300.tiff
                   │   └─    ...
                   │   └─ 5IT-4X_Ch2_z1300.tiff
                   ├─vessel
                   │   └─ 5IT-4X_Ch3_z0300.tiff
                   │   └─    ...
                   │   └─ 5IT-4X_Ch3_z1300.tiff
                   │─virus
                       └─ 5IT-4X_Ch1_z0300.tiff
                       └─    ...
                       └─5IT-4X_Ch1_z1300.tiff


    ```


* In this example the root directory is called ‘3D_Tumor_Lightsheet_Analysis_Pipeline’.  

* The lightsheet data is placed in a folder called ‘data’ and datasets can be organized by study name. For instance, in this example, ‘5IT’ is a folder containing a lightsheet dataset for a single tumor. Within this folder there are sub-folders with tiff files for each of the three fluorescence channels detecting a marker of interest.  

    * **tumor**:  this sub-folder has images with a cell nuclei stain such as syto16, which will be used to delineate the tumor boundary.  

    * **vessel**: this sub-folder has images obtained following immunohistochemical labeling using a marker such as CD31, which will be used to detect blood vessels.  

    * **virus**: this sub-folder has images obtained following immunohistochemical labeling using a marker to detect the biologic of interest. In our use case the biologic of interest is an oncolytic virus, and therefore the sub-folder is named as such. 

* Each tiff file represents a single z plane within a specific marker/fluorescence channel. Therefore, it is recommended that file names incorporate information on study name, channel, and z plane information. E.g., 5IT-4X_Ch1_z0001, where: 

    * ‘5IT’ represents study name 

    * ‘4X’ represents imaging objective or can be other miscellaneous information. 

    * ‘Ch1’ represents the fluorescence channel 

    * z0001 represents the z plane 

* Tiff files for each channel should have the same pixel resolution, format, and dimensions.  

* **NOTE**: The above recommended folder structure is required only when using the fully automated operation mode of the analysis pipeline. However, we recommend utilizing this structure as a best practice even when using the code in a manual mode of operation where individual modules may be run separately allowing further customization. 

#### C) Creation of a configuration (.json file): 

* A configuration file with a **.json** file extension needs to be included in the dataset folder, in order to provide input on data acquisition and analysis parameters. These parameters incude details such as voxel size, choice of segmentation method (e.g., thresholding vs machine-learning based) among others.  

* A detailed list of relevant parameters that need to be specifed in the configuration file are listed below, followed by an example of the .json file itself.  

* [See table of arguments](config.md).


#### D) Hardware Setting

The analysis was performed on the high perfomance computing (HPC) environment. Due to the nature of the code (storing all middle steps localy) it may be used on local computer as well.

##### 1) Our HPC recources

HPC hardware we had at disposal is Cray CS Storm has following recources:

8 GPU Nodes. Each node in the following configuration:

* 8x V100 SXM2 32GB HBM2, NVLink 2
* 2x CLX 6240, 18c, 2.6 GHz (150W)
* 24x 32 GiB DDR4-2933; 768 GiB total
* 4x P4510, NVMe SSD, 2.5”, 2 TB
* 2x S4510, SATA SSD, 2.5”, 240 GB
* 4x Mellanox CX-4, x8, VPI Single-Port, QSFP28

On this setting we were able to analyze one study (containing roughly 1500 Z-planes) in 2 hour. (including resizing preprocessing)

##### 2) Study Parameters

One Lightsheet study has roufly 1500 z-plane images. Each image having resolution of 5732 x 6078 pixels (35 MB each). 
Therefore, one study has 3 x 1500 images requiring almost 1TB storage. 
For the purposes of analysis we resize the images to smaller ones and convert original tiff files to vector arrays.

To operate the code on local computers we recommend the following MINIMAL Hardware Requirements:

##### 3) Minimal Recommended setting for local usage

Based on the input data we recommend following minimal recources:
* CPU with 8 Cores
* 16 GB RAM
* 1TB (hard disk) Storage for the Data (for raw data storage)
* GPU is only required when deep learning model (UNET) is being used.