# GPU Maxmimum Independent Set Solver

This application is a optimization algorithm. It is developed as part of a bachelor thesis at Heidelberg University.

## Dependencies

Kokkos (https://github.com/kokkos/kokkos): Enables performance portable parallelism. Used for running the algorithms on GPUs. \
Cuda (https://developer.nvidia.com/cuda-toolkit): Required to build project. 

## Usage

### Clone the repository

Use `git clone https://github.com/ewaldherr/bthesis` to clone the repository

### Building

Use `-mkdir build` followed by `cd build` to set up the building process. \
Use `cmake .. -DKokkos_ROOT=<your-kokkos-root-directory>`followed by `make` to finish the building process. 

### Algorithms

#### LUBY 

Solves the problem based on Luby's algorithm.

#### LUBYITER

Solves the problem based on Luby's algorithm. Removes vertices from the solution at random, then finding another solution. This process is repeated over many iterations while the best solution is saved.

#### DEGREE

Solves the problem similiar to Luby's algorithm, while including vertices with the lowest degree with higher priority. 

#### DEGREEITER

Solves the problem similiar to Luby's algorithm, while including vertices with the lowest degree with higher priority. Removes vertices from the solution at random, then finding another solution. This process is repeated over many iterations while the best solution is saved.

### Input Format

Input graph needs to be provided in the METIS graph format

### Output Format

The project outputs a file that writes a 1 in line i if vertex i is inside of the found solution, 0 otherwise.