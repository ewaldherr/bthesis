#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include "output.cpp"

// Function to initialize random priorities on the GPU
KOKKOS_FUNCTION void initializePriorities(Kokkos::View<double*> priorities) {
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    Kokkos::parallel_for("init_priorities", priorities.extent(0), KOKKOS_LAMBDA(int i) {
        auto generator = random_pool.get_state();
        priorities(i) = generator.drand(0., 1.);
    });
}

KOKKOS_FUNCTION void checkMax(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, Kokkos::View<double*> priorities, Kokkos::View<int*> state){
    Kokkos::parallel_for("select_max_priority", xadj.extent(0)-1, KOKKOS_LAMBDA(int u) {
            if (state(u) != -1) return;

            bool isMaxPriority = true;
            for (int v = xadj(u); v < xadj(u+1); ++v) {
                if (state(adjncy(v)) == -1 && priorities(adjncy(v)) >= priorities(u)) {
                    isMaxPriority = false;
                    break;
                }
            }

            if (isMaxPriority) {
                state(u) = 2;
            }
        });
}

KOKKOS_FUNCTION void removeVertices(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, Kokkos::View<int*> state){
    Kokkos::parallel_for("update_sets", xadj.extent(0)-1, KOKKOS_LAMBDA(int u) {
        if (state(u) == 2) {
            state(u) = 1;
            for (int v = xadj(u); v < xadj(u+1); ++v) {
                state(adjncy(v)) = 0;
            }
        }
    });
}

// Luby's Algorithm with Kokkos
Kokkos::View<int*> lubysAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy) {
    Kokkos::View<int*> state("state", xadj.extent(0)-1);
    Kokkos::View<double*> priorities("priorities", xadj.extent(0)-1);

    auto h_priorities = Kokkos::create_mirror_view(priorities);
    auto h_state = Kokkos::create_mirror_view(state);
    Kokkos::deep_copy(state, -1);

    bool changes;
    do {
        // Step 1: Assign random priorities to remaining vertices
        initializePriorities(priorities);

        // Step 2: Select vertices with highest priority in their neighborhood
        checkMax(xadj,adjncy,priorities,state);

        // Check if changes occured during last step
        Kokkos::deep_copy(h_state,state);
        changes = false;
        for(int i = 0; i < state.extent(0);++i){
            if(h_state(i)==2){
                changes = true;
                break;
            }
        }
        // Step 3: Add selected vertices to MIS and remove them and their neighbors
        removeVertices(xadj,adjncy,state);

    } while (changes);

    return state;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        //Initialize graph
        int V = 6;
        Kokkos::View<int*> xadj;
        Kokkos::View<int*> adjncy;
        if(argc == 0){
            throw std::runtime_error("No input file. Abort program.");
        }
        else{
            readGraphFromFile(argv[0], xadj,adjncy);
            // Run Luby's algorithm with Kokkos and write results to file
            writeIndependentSetToFile(lubysAlgorithm(xadj,adjncy),"result_mis.txt");
        }
    }

    Kokkos::finalize();
    return 0;
}
