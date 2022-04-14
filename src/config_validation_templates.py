"""
Templates from Pydantic library to validate config file.
"""


# pylint: disable=C0301, E0213, R0201, R0903, C0115

from typing import Dict, Union

from pydantic import BaseModel, DirectoryPath, validator


class MethodWithParametersTemplate(
    BaseModel
):  # pylint: disable=too-few-public-methods, C0301
    """
    Temlate to validate duo method and its parameters.
    """

    method: str
    method_parameters: dict


class ConfigTemplate(BaseModel):  # pylint: disable=too-few-public-methods
    """
    Temlate to validate how intial config should look like.
    """

    segmentation_method_tumor: MethodWithParametersTemplate
    segmentation_method_vessel: MethodWithParametersTemplate
    segmentation_postprocessing_tumor: Union[
        None, MethodWithParametersTemplate
    ]
    segmentation_postprocessing_vessel: Union[
        None, MethodWithParametersTemplate
    ]

    @validator("segmentation_method_vessel")
    def segmented_vessel_model(cls, segmentation_method_vessel):
        """
        test if segmentation method of blood vessels is supported
        """
        supported_list = ["random_forest", "unet", "thresholding"]
        if segmentation_method_vessel.method not in supported_list:
            raise ValueError(
                f"====================================================== \n Segmentation method for vessel must be in {supported_list}, in your config file you have provided {segmentation_method_vessel.method}"
            )
        return segmentation_method_vessel

    @validator("segmentation_method_tumor")
    def segmented_tumor_model(cls, segmentation_method_tumor):
        """
        test if segmentation method of tumor is supported
        """
        supported_list = ["thresholding"]
        if segmentation_method_tumor.method not in supported_list:
            raise ValueError(
                f"====================================================== \n Segmentation method for vessel must be in {supported_list} , in your config file you have provided {segmentation_method_tumor.method}"
            )
        return segmentation_method_tumor

    @validator("segmentation_postprocessing_tumor")
    def mask_postprocessing(cls, segmentation_postprocessing_tumor):
        """
        test if postprocessed functions are supported
        """

        supported_list = ["split_tumor_into_core_and_periphery"]
        if segmentation_postprocessing_tumor.method not in supported_list:
            raise ValueError(
                f"====================================================== \n Segmentation method for vessel must be in {supported_list} , in your config file you have provided {segmentation_postprocessing_tumor.method}"
            )

        return segmentation_postprocessing_tumor

    class Config:
        validate_assignment = True

    data: Dict[str, Dict[str, Dict[str, DirectoryPath]]]
