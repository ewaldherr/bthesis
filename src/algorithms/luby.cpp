#include "reductions.cpp"

// Function to initialize random priorities on the GPU
KOKKOS_FUNCTION void initializePriorities(Kokkos::View<double*>& priorities, unsigned int seed) {
    Kokkos::parallel_for("init_priorities", priorities.extent(0), KOKKOS_LAMBDA(int i) {
        // Directly create a random number generator for each thread
        Kokkos::Random_XorShift64<Kokkos::DefaultExecutionSpace> generator(seed + i);
        // Generate a random number and assign it to the priorities view
        priorities(i) = generator.drand(0., 1.);
    });
}


// Function that checks for each vertex if it has the max priority of its neighborhood
KOKKOS_FUNCTION int checkMax(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<double*>& priorities, Kokkos::View<int*>& state){
    int changes = 0;
    Kokkos::parallel_reduce("select_max_priority", xadj.extent(0)-1, KOKKOS_LAMBDA(int u, int& vertices) {
            if (state(u) != -1) return;

            bool isMaxPriority = true;
            for (int v = xadj(u); v < xadj(u+1); ++v) {
                if ((state(adjncy(v)) == -1 && priorities(adjncy(v)) >= priorities(u)) || state(adjncy(v)) == 1) {
                    isMaxPriority = false;
                    break;
                }
            }

            if (isMaxPriority) {
                state(u) = 1;
                for (int v = xadj(u); v < xadj(u+1); ++v) {
                    state(adjncy(v)) = 0;
                    vertices++;
                }
            }
        }, changes);
    return changes;
}

// Luby's Algorithm
Kokkos::View<int*> lubysAlgorithm(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& state, unsigned int seed) {
    Kokkos::View<double*> priorities("priorities", xadj.extent(0)-1);

    // Assign random priorities to remaining vertices
    initializePriorities(priorities, seed);
    //int totalIterations = 0;
    int pool = xadj.extent(0) - 1;
    int changed;
    bool changes;
    do {
        changed = checkMax(xadj,adjncy,priorities,state);
        pool -= changed;
        std::cout << pool << " vertices are left" << std::endl; 
        changes = (changed > 0);
        //++totalIterations;
    } while (changes);

    //std::cout << "The algorithm run a total of " << totalIterations << " total iterations" << std::endl;
    return state;
}

