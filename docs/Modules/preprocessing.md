# Preprocessing

::: src.preprocessing

This step will transform and downsize input data (**tumor**, **vessel**, **virus**) *.tiff* files into python's numpy array files, which will be saved to the output path directory. The new folder structure will look as follows:

```bash
ppdm
└─ data
   └─ 5IT_STUDY
        └─ config.json
        └─ source
            └─raw
            │   └─tumor
            │   │   └─ 5IT-4X_Ch2_z0300.tiff
            │   │   └─    ...
            │   │   └─ 5IT-4X_Ch2_z1300.tiff
            │   ├─vessel
            │   │   └─ 5IT-4X_Ch3_z0300.tiff
            │   │   └─    ...
            │   │   └─ 5IT-4X_Ch3_z1300.tiff
            │   │─virus
            │       └─ 5IT-4X_Ch1_z0300.tiff
            │       └─    ...
            │       └─5IT-4X_Ch1_z1300.tiff
------------│-------------------------------------------------------  
            └─transformed
                   └─ np_and_resized
                           └─tumor
                           │   └─ 5IT-4X_Ch2_z0300.np
                           │   └─    ...
                           │   └─ 5IT-4X_Ch2_z1300.np
                           ├─vessel
                           │   └─ 5IT-4X_Ch3_z0300.np
                           │   └─    ...
                           │   └─ 5IT-4X_Ch3_z1300.np
                           │─virus
                               └─ 5IT-4X_Ch1_z0300.np
                               └─    ...
                               └─5IT-4X_Ch1_z1300.np
```

