This directory refers to the chapter 5 of the report.

Run ```post_processing_generator.py``` first then run ```post_processing_average.py``` to perform the averaging and generate the alternative dataset  

The alternative datasets are registered in directory ```post_processing_data/ppTraining```. The same dataset withouts post processing is registered in ```post_processing_data/training```. Generated datasets can be provided to neural networks.

Number of datasets and number of trajectories per dataset can be modified in the file ```post_processing_generator.py```.

Variables  discussed in the report ```n``` and ```distance_exclusion_value``` can be mofdified in the file ```post_processing_average.py```. 