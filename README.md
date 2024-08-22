# AmpSuite: Integrated Amplitude, Coda, Discrimination and Yield Routines

AmpSuite is a comprehensive toolset for seismic data analysis, including amplitude measurement, correction, discrimination, and yield estimation.

## Installation

### Setting up the environment

1. Create the conda environment:
   ```
   conda env create -f environment-nocuda.yml
   ```

### Installing AmpSuite package

1. Activate the conda environment:
   ```
   conda activate ampsuite
   ```

2. Install the package in editable mode:
   ```
   pip install -e .
   ```

## Tutorials

Explore the following Jupyter notebooks to learn how to use AmpSuite:

1. [Measure Amplitudes](notebooks/Tutorial001_Measure_Amplitudes.ipynb)
2. [Correct Amplitudes](notebooks/Tutorial002_Correct_Amplitudes.ipynb)
3. [Discrimination Analysis (No Source Fit)](notebooks/Tutorial003_Discrimination_Analysis_No_Source_Fit.ipynb)
4. [Simple Bayesian Yield Estimation](notebooks/Tutorial004_Bayesian_Yield_Estimation.ipynb)
5. [Plot Amplitude Tomography](notebooks/Tutorial005_Plot_Amplitude_Tomography.ipynb)
6. [Gaussian Process Yield Estimation](notebooks/Tutorial006_Gaussian_Process_Yield_Estimation.ipynb)

## Contributing

Contributions to AmpSuite are welcome! Please refer to our [contributing guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the [MIT License](LICENSE.md).

## Contact

For questions or support, please [open an issue](https://github.inl.gov/richard-alfaro-diaz-lanl/AmpSuite/issues) on our GitHub repository.