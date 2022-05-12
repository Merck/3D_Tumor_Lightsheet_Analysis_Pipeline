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