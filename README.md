## Project Overview

The exercises focus on:
1. Generating a handwritten '8' using a fully connected neural network.
2. Denoising noisy images of the digit '8' using gradient descent optimization.
3. Reconstructing reduced-resolution images of the digit '8' while adding noise.

### Structure of the Exercises

1. **Problem 2.1** - A generative model creates a handwritten '8' using two fully connected layers with ReLU and sigmoid activations. The input is a random Gaussian vector.
2. **Problem 2.2** - Reconstructing noisy images of a '8' by retaining a subset of pixel values.
3. **Problem 2.3** - Reconstructing images with reduced resolution and added noise using matrix transformations.

## Files in this Repository

- `data21.mat`, `data22.mat`, `data23.mat`: MATLAB data files containing matrices and vectors used for the neural network transformations and image reconstructions.
- `ex1.m`, `ex2.m`, `ex3.m`: MATLAB scripts corresponding to each problem.

## Usage Instructions

1. **Setup**
   - Ensure MATLAB is installed on your system.
   - Load the data files using the scripts provided (e.g., `load('data21.mat')`).

2. **Running the Scripts**
   - Each script (`ex1.m`, `ex2.m`, `ex3.m`) corresponds to a specific problem in the exercise.
   - You can modify input parameters (e.g., `N` values for pixel retention) as described in the code to observe different results.

3. **Visualization**
   - The generated images, denoised images, and reconstructed images will be displayed using MATLAB's `imshow` function.


## Results
