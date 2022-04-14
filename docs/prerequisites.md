#### PREREQUISITES

The code expects to have :

* Python installed on HPC (e.g. via minoconda)
* Python virtual environment containing all required packages and modules (3d environment)

In order to install infrastructure on the hpc server (python, packages, etc.) follow [this guide](installation on HPC and tips.md).



#### Folder Structure

Due to the size of the data, every single step dumps the results on the disk (as a substitute for the RAM). Therefore, the code operates with a fixed folder structure

* When analyzing one particular study (*5_IT_STUDY* in this case), the following **folders structure** of three channels (vessels, tumors, virus) including the **config.json** file is expected.

* Expected Folder Structure

    The **root directory** is considered the **ppdm** folder (see tree diagram below)

    ```bash
    ppdm
    └─ data
       └─ 5IT_STUDY
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

* *.tiff* files are expected to be exported by Imaris software. We expect that all the three channels have the same format (one channel *tiff* file) of the same size.

**NOTE**: This folder structure is expected only for the **Fully automated pipeline** usuage. However, we recommend uttilizing this structure even when using the code in a **Manual Usage of Modules**.