### Code use pre-requisites and Installation instructions

Prior to running the tumor lightsheet image analysis pipeline, the following pre-requisites must be met: 

* A) Installation of the python environment named ‘3d’ that contains all the necessary python packages and modules required to run this code.  
* B) Organization of lightsheet data in the recommended folder architecture 
* C) Creation of a configuration .json file 


Each of these pre-requisites are described in detail below.  

#### A) Installation of the required python environment:  

* Guide for the installation [can be found at this link](install.md)

#### B)Folder Structure

Due to the large size of lightsheet datasets, the code is written so as to allow output from each step in the analysis pipeline to directly save results on the hard disk. This overcomes the limitation of limited RAM. As a consequence, it is necessary to use a fixed folder structure for the raw data and configuration file provided as an input. The recommended folder structure is illustrated with an example below: 

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

* The lightsheet data is placed in a folder called ‘data’ and can datasets can be organized by study name. For instance, in this example, ‘5IT’ is a folder containing a lightsheet dataset for a single tumor. Within this folder there are sub-folders with tiff files for each of the three fluorescence channels detecting a marker of interest.  

    * **tumor**:  this sub-folder has images with a cell nuclei stain such as syto16, which will be used to delineate the tumor boundary.  

    * **vessel**: this sub-folder has images obtained following immunohistochemical labeling using a marker such as CD31, which will be used to detect blood vessels.  

    * **virus**: this sub-folder has images obtained following immunohistochemical labeling using a marker to detect the biologic of interest. In our use case the biologic of interest is an oncolytic virus, and therefore the sub-folder is named as such. 

* Each tiff file represents a single z plane for within a specific marker/fluorescence channel. Therefore, it is recommended that file names incorporate information on study name, channel, and z plane information. E.g., 5IT-4X_Ch1_z0001, where: 

    * ‘5IT’ represents study name 

    * ‘4X’ represents imaging objective or can be other miscellaneous information. 

    * ‘Ch1’ represents the fluorescence channel 

    * z0001 represents the z plane 

* Tiff files for each channel should have the same pixel resolution, format, and dimensions.  

* **NOTE**: The above recommended folder structure is required only when using the fully automated operation mode of the analysis pipeline. However, we recommend uttilizing this structure as a best practice even when using the code in a manual mode of operation where individual modules may be run separately allowing further customization. 

#### C) Creation of a configuration (.json file): 

* A configuration file with a ‘.json’ file extension needs to be included in the dataset folder, in order to provide input on data acquisition and analysis parameters. These parameters incude details such as voxel size, choice of segmentation method (e.g., thresholding vs machine-learning based) among others.  

* A detailed list of relevant parameters that need to be specifed in the configuration file are listed below, followed by an example of the .json file itself.  

* [See table of arguments](config.md).