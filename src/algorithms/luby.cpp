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
KOKKOS_FUNCTION void checkMax(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<double*>& priorities, Kokkos::View<int*>& state){
    Kokkos::parallel_for("select_max_priority", xadj.extent(0)-1, KOKKOS_LAMBDA(int u) {
            if (state(u) != -1) return;

            bool isMaxPriority = true;
            for (int v = xadj(u); v < xadj(u+1); ++v) {
                if ((state(adjncy(v)) == -1 && priorities(adjncy(v)) >= priorities(u)) || state(adjncy(v)) == 2) {
                    isMaxPriority = false;
                    break;
                }
            }

            if (isMaxPriority) {
                state(u) = 2;
                for (int v = xadj(u); v < xadj(u+1); ++v) {
                    state(adjncy(v)) = 0;
                }
            }
        });
}

// Function to remove vertices of vertices added to MIS
KOKKOS_FUNCTION void removeVertices(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& state){
    Kokkos::parallel_for("update_sets", xadj.extent(0)-1, KOKKOS_LAMBDA(int u) {
        if (state(u) == 2) {
            state(u) = 1;
            for (int v = xadj(u); v < xadj(u+1); ++v) {
                state(adjncy(v)) = 0;
            }
        }
    });
}

KOKKOS_FUNCTION bool isDone(Kokkos::View<int*>& state){
    int sum = 0;
    Kokkos::parallel_reduce("count_unassigned", state.extent(0), KOKKOS_LAMBDA(const int i, int& vertices) {
        if(state(i) == -1){
            vertices++;
        }
    }, sum);
    if(sum > 0){
        return true;
    }
    return false;
}

// Luby's Algorithm
Kokkos::View<int*> lubysAlgorithm(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& state, unsigned int seed) {
    Kokkos::View<double*> priorities("priorities", xadj.extent(0)-1);

    auto h_state = Kokkos::create_mirror_view(state);

    // Assign random priorities to remaining vertices
    initializePriorities(priorities, seed);
    int totalIterations = 0;

    bool changes;
    do {

        // Select vertices with highest priority in their neighborhood
        checkMax(xadj,adjncy,priorities,state);

        // Check if changes occured during last step


        // Add selected vertices to MIS and remove them and their neighbors
        //removeVertices(xadj,adjncy,state);
        changes = isDone(state);

        totalIterations++;
    } while (changes);

    std::cout << "The algorithm run a total of " << totalIterations << " total iterations" << std::endl;
    return state;
}

