#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <string>
#include <ctime>
#include <chrono>

// Function to initialize random priorities on the GPU
KOKKOS_FUNCTION void initializePriorities(Kokkos::View<double*>& priorities, unsigned int seed) {
    Kokkos::Random_XorShift64_Pool<> random_pool(seed);
    Kokkos::parallel_for("init_priorities", priorities.extent(0), KOKKOS_LAMBDA(int i) {
        auto generator = random_pool.get_state();
        priorities(i) = generator.drand(0.,1.);
        random_pool.free_state(generator);
    });
}

// Function that checks for each vertex if it has the max priority of its neighborhood
KOKKOS_FUNCTION void checkMax(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<double*>& priorities, Kokkos::View<int*>& state) {
    Kokkos::parallel_for("select_max_priority", xadj.extent(0) - 1, KOKKOS_LAMBDA(int u) {
        if (state(u) != -1) return;

        // Check bounds for xadj
        if (u + 1 >= xadj.extent(0)) return;

        bool isMaxPriority = true;
        for (int v = xadj(u); v < xadj(u+1); ++v) {
            // Check bounds for adjncy
            if (v >= adjncy.extent(0)) continue;
            int neighbor = adjncy(v);
            // Validate neighbor index
            if (neighbor < 0 || neighbor >= state.extent(0)) continue;

            if ((state(neighbor) == -1 && priorities(neighbor) >= priorities(u)) || state(neighbor) == 2) {
                isMaxPriority = false;
                break;
            }
        }

        if (isMaxPriority) {
            state(u) = 2;
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

// Luby's Algorithm
Kokkos::View<int*> lubysAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, Kokkos::View<int*> state, unsigned int seed) {
    Kokkos::View<double*> priorities("priorities", xadj.extent(0)-1);

    auto h_state = Kokkos::create_mirror_view(state);
    Kokkos::deep_copy(state, -1);

    std::cout << "initialized host copies" << std::endl;
    // Assign random priorities to remaining vertices
    initializePriorities(priorities, seed);
    std::cout << "initialized priorities" << std::endl;
    bool changes;
    do {

        // Select vertices with highest priority in their neighborhood
        checkMax(xadj,adjncy,priorities,state);
        std::cout << "Checked max" << std::endl;
        // Check if changes occured during last step
        Kokkos::deep_copy(h_state,state);
        changes = false;
        for(int i = 0; i < state.extent(0);++i){
            if(h_state(i)==2){
                changes = true;
                break;
            }
        }
        std::cout << "Checked for changes" << std::endl;
        // Add selected vertices to MIS and remove them and their neighbors
        removeVertices(xadj,adjncy,state);
        std::cout << "Removed vertices" << std::endl;
    } while (changes);

    return state;
}

