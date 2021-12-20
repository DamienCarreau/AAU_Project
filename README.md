# AAU_Project

This directory contains the code related to the research project "Relevance of a neural network inplace of a model predictive controller" whose pdf is on git.

The file ```generator.py``` allows to generate or regenerate datasets. Attention, in the folder ```data/training``` training files are already present. Running the file will rewrite them.

The ```cross_validation.py``` file will run the cross-validation method, described in section 4.4 of the report, on different neural networks. The parameters of these networks can be modified in the file. If a network performs well with a percentage of valid predictions above 80%, it will be saved in the ```data/models``` folder. The results are saved in the file ```data/validation/validation.txt```.

The ```time_efficiency.py``` file will measure for some efficient models of different types their time efficiency on different channels. It will also measure the prediction time of the ACADO Toolkit MPC. The results will be stored in the file ```data/validation/efficency.txt```.
<hr>

To install the ```acado``` module in python, refer to the file ```mpc_acado/setup.py```.
