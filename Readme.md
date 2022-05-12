## Description

This repo provides code for **3D Histology applications for drug discovery** paper.

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





***Title: <br />
High-Resolution ex vivo tissue clearing, lightsheet imaging, and data analysis to support macromolecular drug and biomarker distribution in whole organs and tumors. <br /> <br /> Short Tile:<br />
3D Histology applications for drug discovery.***


**Authors:<br />**

Niyanta Kumar, Petr Hrobař, Martin Vagenknecht, Jindrich Soukup, Peter Bloomingdale, Tomoko Freshwater, Sophia Bardehle, Roman Peter, Nadia Patterson, Ruban Mangadu, Cinthia Pastuskovas, and Mark Cancilla
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

Firtsly Download the REPO (Entire project in one folder) by running:
```bash
# Clone The repo localy to your computer
git clone https://github.com/Merck/3D_Tumor_Lightsheet_Analysis_Pipeline.git

# Navigate to the repo folder
cd 3D_Tumor_Lightsheet_Analysis_Pipeline
```

### 2) **Dependencies Installation**:

When installing the package:

* **MAC/LINUX Users**

    2.1) Make sure you have a python installed on your computer.
    if not you may use `set_python.sh` file
    Run in the terminal:
    ```bash
    source environment_setup/set_python.sh

    ```

    2.2) Create a python environment  
    Run in the terminal:
    ```bash
    source environment_setup/set_3d_infrastructure.sh
    ```

* **Windows Users**

    2.1) Make sure you have a python installed on your computer.
    if not, you may use [this link.](https://docs.conda.io/en/latest/miniconda.html)

    2.2) Create a python environment run all lines of `environment_setup/set_3d_infrastructure.sh` manually in the terminal


# 

### **Copyright**
Copyright © 2022 Merck & Co., Inc., Kenilworth, NJ, USA and its affiliates. All rights reserved.


# 

### **Refs**
[^1]: Dobosz, M., Ntziachristos, V., Scheuer, W. & Strobel, S. **Multispectral Fluorescence Ultramicroscopy: Three-Dimensional Visualization and Automatic Quantification of Tumor Morphology, Drug Penetration, and Antiangiogenic Treatment Response**. Neoplasia 16, 1-U24, doi:10.1593/neo.131848 (2014).*


