# clone project

# go to repo root (3D_Tumor_Lightsheet_Analysis_Pipeline)


# create python environment
conda env create --file=environment.yml -y
# activate the environment
conda activate 3d
# go to correct branch (won"t be needed in the future)
#git checkout code_refactoring
# install pytorch
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install fastai==2.5.1 fastbook==0.0.18 fastcore==1.3.26 fastprogress==1.0.0

# create jupter lab kernel
python -m ipykernel install --user --name 3d
# install project as a python library
pip install -e .

# For some Jupyter Lab extensions you might need nodejs
# Jupyter Lab extension itkwidgets (3d visualization library)
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib jupyterlab-datawidgets itkwidgets

# Jupyter Lab extension Jupytext - optional
jupyter labextension install jupyterlab-jupytext@1.2.2  # For Jupyter Lab 2.x
