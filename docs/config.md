Config file contains an entire setting for the automated analysis via **main.py** file and **master_script.py**. It allows changing methods for blood vessels segmentation as well as changing pre-trained models to use.


#### config.json file
All Changeble parameters are listed bellow:

| Main Key                      | method/value | method_parameters                              |   |   |
|-------------------------------|-------------------|----------------------------------------|---|---|
| **segmentation_method_tumor[^1]** | **thresholding[^2]**  | **method**: [th_triangle,th_yen,th_otsu]   |   |   |
| **segmentation_method_vessel[^1]** | **random_forest[^2]** | **model_file**: relative path              |   |   |
|                               | **unet[^2]**          | **model_file**: relative path              |   |   |
|                               | **thresholding[^2]**  | **method**: [th_triangle,th_yen,th_otsu]   |   |   |
| **segmentation_postprocessing_tumor** | **split_tumor_into_core_and_periphery[^3]** | **periphery_as_ratio_of_max_distance**:<0; 1>               |   |   |
| **distance_tranform** |  | **stack_size**: int              |   |   |
| **pixels_to_microns** | float  |               |   |   |
| **mlflow_logging** | bool  |               |   |   |
| **mlflow_run_name** | str  |               |   |   |

[^1]: [see pipeline overview](methodology_overview.md)
[^2]: [see segmentation module documentation and examples](Modules/segmentation.md)
[^3]: Describes the fraction of tumor core and periphery (should add up to 1) - e.g. we assumed 0.2 (20 percent) periphery and 0.8 (80 percent) tumor core.

# 

#### Example of possible config.json file.

```bash
{
    "segmentation_method_tumor": {  
        "method": "thresholding",          # Select segmentation method for tumor channel (here "thresholding")
        "method_parameters": {
            "method": "th_triangle"        # What thresholding method to use: yen, triangle, otsu...
        }
    },
    "segmentation_method_vessel": {  
        "method": "unet",                 # What method to use for blood vessels segmentation
        "method_parameters": {
            "model_file": "ppdm/data/unet_model.pt", # path to the pre-trained model
        }
    },
    "segmentation_postprocessing_tumor": {               # Tumor postprocesing - splitting the brains to core and periphery
        "method": "split_tumor_into_core_and_periphery", # Split tumor to the core and periphery
        "method_parameters": {}                          # No parameter necessary for this method - defaults to 0.2 (20 percent core, 80 periphery)
    },
    "distance_tranform": {                      # (Outer) Distance Transform for the blood vessels distance
        "method_parameters": {                  # How many layers stacked together inside DT aggregation
            "stack_size": 100
        }
    }
    "pixels_to_microns": 4,     # Multiplication constant for converting pixels to micrones
    "mlflow_logging": true      # Bool parameters if the results should be saved to mlflow.
    "mlflow_run_name": "DEMO"   # Name of the experiment which should be used for the logging
}

```

