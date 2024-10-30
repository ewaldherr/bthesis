#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>

// Function to initialize random priorities on the GPU
KOKKOS_FUNCTION void initializePriorities(Kokkos::View<double*> priorities) {
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/1234);
    Kokkos::parallel_for("init_priorities", priorities.extent(0), KOKKOS_LAMBDA(int i) {
        auto generator = random_pool.get_state();
        priorities(i) = generator.drand(0., 1.);
    });
}

KOKKOS_FUNCTION void checkMax(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, Kokkos::View<double*> priorities, Kokkos::View<int*> state){
    Kokkos::parallel_for("select_max_priority", xadj.extent(1)-1, KOKKOS_LAMBDA(int u) {
            if (state(u) != 0) return;

            bool isMaxPriority = true;
            for (int v = xadj(u); v < xadj(u+1)-1; ++v) {
                if (state(v) == 0 && priorities(adjncy(u+v)) >= priorities(u)) {
                    isMaxPriority = false;
                    break;
                }
            }

            if (isMaxPriority) {
                state(u) = 1;
            }
        });
}

KOKKOS_FUNCTION void removeVertices(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, Kokkos::View<int*> state){
    Kokkos::parallel_for("update_sets", xadj.extent(1)-1, KOKKOS_LAMBDA(int u) {
        if (state(u) == 1) {
            state(u) = 2;
            for (int v = xadj(u); v < xadj(u+1)-1; ++v) {
                state(adjncy(u+v)) = -1;
            }
        }
    });
}

// Luby's Algorithm with Kokkos
Kokkos::View<int*> lubysAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy) {
    Kokkos::View<int*> state("state", xadj.extent(1)-1);
    Kokkos::View<double*> priorities("priorities", xadj.extent(1)-1);

    auto h_state = Kokkos::create_mirror_view(state);
    Kokkos::deep_copy(state, 0);

    bool changes;
    do {
        // Step 1: Assign random priorities to remaining vertices
        initializePriorities(priorities);
        // Step 2: Select vertices with highest priority in their neighborhood
        checkMax(xadj,adjncy,priorities,state);
        Kokkos::deep_copy(h_state,state);

        // Check if changes occured during last step
        changes = false;
        for(int i = 0; i < state.extent(0);++i){
            if(h_state(i)==1){
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
        Kokkos::View<int*> xadj ("xadj",V+1);
        Kokkos::View<int*> adjncy ("adjncy",V*V);
        auto h_xadj = Kokkos::create_mirror_view(xadj);
        auto h_adjncy = Kokkos::create_mirror_view(adjncy);
        h_xadj(0) = 0;
        h_xadj(1) = 2;
        h_xadj(2) = 4;
        h_xadj(3) = 6;
        h_xadj(4) = 10;
        h_xadj(5) = 11;
        h_xadj(6) = 12;
        h_adjncy(0) = 1;
        h_adjncy(1) = 2;
        h_adjncy(2) = 0;
        h_adjncy(3) = 3;
        h_adjncy(4) = 0;
        h_adjncy(5) = 3;
        h_adjncy(6) = 1;
        h_adjncy(7) = 2;
        h_adjncy(8) = 4;
        h_adjncy(9) = 5;
        h_adjncy(10) = 3;
        h_adjncy(11) = 3;

        Kokkos::deep_copy(adjncy,h_adjncy);
        Kokkos::deep_copy(xadj,h_xadj);
        // Run Luby's algorithm with Kokkos
        Kokkos::View<int*> independentSet = lubysAlgorithm(xadj,adjncy);
        // Print the result
        auto h_set = Kokkos::create_mirror_view(independentSet);
        Kokkos::deep_copy(h_set,independentSet);
        std::cout << "Maximum Independent Set (MIS) nodes:" << std::endl;
        for (int i = 0; i < h_set.extent(0); ++i){
            if (h_set(i) == 2) {
                std::cout << i << " ";
            }
        }
        std::cout << std::endl;
    }

    Kokkos::finalize();
    return 0;
}
