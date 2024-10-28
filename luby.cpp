#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>

// Function to initialize random priorities on the GPU
KOKKOS_FUNCTION void initializePriorities(Kokkos::View<int*> priorities) {
    Kokkos::parallel_for("init_priorities", priorities.extent(0), KOKKOS_LAMBDA(int i) {
        unsigned seed = 1234 + i; // Seed for random number generator
        priorities(i) = rand_r(&seed);
    });
}

// Luby's Algorithm with Kokkos
KOKKOS_FUNCTION Kokkos::View<int*> lubysAlgorithm(Kokkos::View<int**> graph) {
    Kokkos::View<int*> state("state", graph.extent(1));
    Kokkos::View<int*> priorities("priorities", graph.extent(1));
    
    Kokkos::deep_copy(state, 0);

    bool changes;
    do {
        // Step 1: Assign random priorities to remaining vertices
        initializePriorities(priorities);

        // Step 2: Select vertices with highest priority in their neighborhood
        Kokkos::parallel_for("select_max_priority", graph.extent(1), KOKKOS_LAMBDA(int u) {
            if (state(u) != 0) return;

            bool isMaxPriority = true;
            for (int v = 0; v < graph.extent(1); ++v) {
                if (graph(u, v) == 1 && state(v) == 0 && priorities(v) >= priorities(u)) {
                    isMaxPriority = false;
                    break;
                }
            }

            if (isMaxPriority) {
                state(u) = 1;
            }
        });

        // Step 3: Add selected vertices to MIS and remove them and their neighbors
        changes = false;
        Kokkos::parallel_reduce("update_sets", graph.extent(1), KOKKOS_LAMBDA(int u, bool &local_changes) {
            if (state(u) == 1) {
                state(u) = 2;
                for (int v = 0; v < graph.extent(1); ++v) {
                    if (graph(u, v) == 1) {
                        state(v) = -1;
                    }
                }
                local_changes = true; // If any vertex is added, flag a change
            }
        }, changes);

    } while (changes);

    return state;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        //Initialize graph
        int V = 6;
        Kokkos::View<int**> adj ("adj",V,V);
        auto h_graph = Kokkos::create_mirror_view(adj);
        h_graph(0, 1) = 1;
        h_graph(0, 2) = 1;
        h_graph(1, 3) = 1;
        h_graph(2, 3) = 1;
        h_graph(3, 4) = 1;
        h_graph(3, 5) = 1;
        Kokkos::deep_copy(adj,h_graph);
        // Run Luby's algorithm with Kokkos
        Kokkos::View<int*> independentSet = lubysAlgorithm(adj);
        // Print the result
        auto h_set = Kokkos::create_mirror_view(independentSet);
        Kokkos::deep_copy(h_set,independentSet);
        std::cout << "Maximum Independent Set (MIS) nodes:" << std::endl;
        for (int i = 0; i < h_set.extent(0); ++i){
            if (h_set(i) == 1) {
                std::cout << i << " ";
            }
        }
        std::cout << std::endl;
    }

    Kokkos::finalize();
    return 0;
}
