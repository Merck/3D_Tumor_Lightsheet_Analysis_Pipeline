## Description

This repo provides code for lightsheet imaging data analysis pipeline for extraction of quantitative readouts regarding vascular volume and drug penetration in tumors.

**A custom data analysis pipeline was developed to enable rapid analysis of tumor lightsheet datasets. Key goals of this analysis pipeline are:**

* Enable extraction of quantitative readouts regarding drug penetration in whole tumors from lightsheet data sets. See Dobosz et al. 2014., Neoplasia[^1] for reference. 
* Use python programming to make use of open source packages that can support building a custom pipeline.
* Use cloud computing environment (Merck High Performance Computing Resources) to enable rapid analysis of very large lightsheet data sets. 
* Develop an analysis pipeline with two run modes:

    a) Automatic run mode: enables executing full analysis pipeline on a new data set via a single line of code

    b) Lego Brick mode: enables re-using parts of the analysis pipeline for building new analysis methods. 

# 
## Code Documentation 
**The code documentation and examples may be found [here](https://merck.github.io/3D_Tumor_Lightsheet_Analysis_Pipeline/)**


#

## Learn more in our publication:





***
***Title***: 
A Light sheet fluorescence microscopy and machine learning-based approach to investigate drug and biomarker distribution in whole organs and tumors. <br /> <br /> ***


**Authors:<br />**

Niyanta Kumar, Petr Hrobař, Martin Vagenknecht, Jindrich Soukup, Nadia Patterson, Peter Bloomingdale, Tomoko Freshwater, Sophia Bardehle, Roman Peter, Ruban Mangadu, Cinthia V. Pastuskovas, and Mark T. Cancilla
<br /><br />***Merck & Co., Inc.***
***


## Development Team

In case of need of any editional (technical) information reach out to the development team via **github's issue**:

* Martin Vagenknech (Merck & Co., Inc.)
* Petr Hrobar (Merck & Co., Inc.)
* Jindrich Soukup (Merck & Co., Inc.)


***

## **Installation Process**

Repo is a python package. Installation process can be automated via bash `.sh` file in the `environment_setup` folder.


### 1) **Code Download**

Clone the github repository (Entire project in one folder) by running:
```bash
# Clone The repo localy to your computer
git clone https://github.com/Merck/3D_Tumor_Lightsheet_Analysis_Pipeline.git

# Navigate to the repo folder
cd 3D_Tumor_Lightsheet_Analysis_Pipeline
```

### 2) **Dependencies Installation**:

When installing the package:

* **MAC/LINUX Users**

    2.1) Make sure you have conda installed on your computer.
    if not, you may use [this link.](https://docs.conda.io/en/latest/miniconda.html)
    
    2.2) Create a python environment  
    Run in the terminal:
    ```bash
    source environment_setup/set_3d_infrastructure.sh
    ```

* **Windows Users**

    2.1) Make sure you have conda installed on your computer.
    if not, you may use [this link.](https://docs.conda.io/en/latest/miniconda.html)

    2.2) Create a python environment run all lines of `environment_setup/set_3d_infrastructure.sh` manually in the terminal

# 
### Hardware Requirements:
To operate the code on local computers we recommend the following MINIMAL Hardware Requirements:

* CPU with at least 6 Cores
* 16 GB RAM
* 800 GB Storage for the Data
* GPU is only required when deep learning model (UNET) is being used.



#
### **Copyright**
Copyright © 2022 Merck & Co., Inc., Kenilworth, NJ, USA and its affiliates. All rights reserved.


# 

### **Refs**
[^1]: Dobosz, M., Ntziachristos, V., Scheuer, W. & Strobel, S. **Multispectral Fluorescence Ultramicroscopy: Three-Dimensional Visualization and Automatic Quantification of Tumor Morphology, Drug Penetration, and Antiangiogenic Treatment Response**. Neoplasia 16, 1-U24, doi:10.1593/neo.131848 (2014).*


