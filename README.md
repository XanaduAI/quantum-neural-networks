<img align="left" src="https://github.com/XanaduAI/quantum-neural-networks/blob/master/static/tetronimo.png" width=300px>

# Continuous-variable quantum neural networks

This repository contains the source code used to produce the results presented in *"Continuous-variable quantum neural networks"* [arXiv:1806.06871](https://arxiv.org/abs/1806.06871).

<br/>

## Contents

<!-- <p align="center">
	<img src="https://github.com/XanaduAI/quantum-neural-networks/blob/master/static/function_fitting.png">
</p> -->

* **Function fitting**: The folder `function_fitting` contains the Python script `function_fitting.py`, which automates the process of fitting classical functions using continuous-variable (CV) variational quantum circuits. Simply specify the function you would like to fit, along with other hyperparameters, and this script automatically constructs and optimizes the CV quantum neural network. In addition, training data is also provided.

* **Quantum autoencoder**: coming soon.

* **Quantum fraud detection**: The folder `fraud_detection` contains the Python script `fraud_detection.py`, which builds and trains a hybrid classical/quantum model for fraud detection. Additional scripts are provided for visualizing the results.

* **Tetronimo learning**: coming soon.

<img align='right' src="https://github.com/XanaduAI/quantum-neural-networks/blob/master/static/tetronimo_gif.gif">


## Requirements

To construct and optimize the variational quantum circuits, these scripts and notebooks use the TensorFlow backend of [Strawberry Fields](https://github.com/XanaduAI/strawberryfields). In addition, matplotlib is required for generating output plots.

**Due to subsequent interface upgrades, these scripts will work only with Strawberry Fields version <= 0.10.0.**


## Using the scripts

To use the scripts, simply set the input data, output data, and hyperparametersby modifying the scripts directly - and then enter the subdirectory and run the script using Python 3:

```bash
python3 script_name.py
```

The outputs of the simulations will be saved in the subdirectory.

To access any saved data, the file can be loaded using NumPy:

```python
results = np.load('simulation_results.npz')
```

## Authors

Nathan Killoran, Thomas R. Bromley, Juan Miguel Arrazola, Maria Schuld, Nicolás Quesada, and Seth Lloyd.

If you are doing any research using this source code and Strawberry Fields, please cite the following two papers:

> Nathan Killoran, Thomas R. Bromley, Juan Miguel Arrazola, Maria Schuld, Nicolás Quesada, and Seth Lloyd. Continuous-variable quantum neural networks. arXiv, 2018. [arXiv:1806.06871](https://arxiv.org/abs/1806.06871)

> Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. Strawberry Fields: A Software Platform for Photonic Quantum Computing. arXiv, 2018. [Quantum, 3, 129](https://quantum-journal.org/papers/q-2019-03-11-129/) (2019).

## License

This source code is free and open source, released under the Apache License, Version 2.0.
