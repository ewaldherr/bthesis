#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>

// Function to initialize random priorities on the GPU
KOKKOS_INLINE_FUNCTION void initializePriorities(Kokkos::View<int*> priorities) {
    Kokkos::parallel_for("init_priorities", priorities.extent(0), KOKKOS_LAMBDA(int i) {
        unsigned seed = 1234 + i; // Seed for random number generator
        priorities(i) = rand_r(&seed);
    });
}

// Luby's Algorithm with Kokkos
KOKKOS_INLINE_FUNCTION Kokkos::View<int*> lubysAlgorithm(Kokkos::view<int**> graph) {
    Kokkos::View<int*> inMIS("inMIS", graph.extend(0));
    Kokkos::View<int*> removed("removed", graph.extend(0));
    Kokkos::View<int*> priorities("priorities", graph.extend(0));

    Kokkos::deep_copy(inMIS, 0);
    Kokkos::deep_copy(removed, 0);

    bool changes;
    do {
        // Step 1: Assign random priorities to remaining vertices
        initializePriorities(priorities);

        // Step 2: Select vertices with highest priority in their neighborhood
        Kokkos::parallel_for("select_max_priority", graph.V, KOKKOS_LAMBDA(int u) {
            if (removed(u) == 1) return;

            bool isMaxPriority = true;
            for (int v = 0; v < graph.V; ++v) {
                if (graph.adj(u, v) == 1 && removed(v) == 0 && priorities(v) >= priorities(u)) {
                    isMaxPriority = false;
                    break;
                }
            }

            if (isMaxPriority) {
                inMIS(u) = 1;
            }
        });

        // Step 3: Add selected vertices to MIS and remove them and their neighbors
        changes = false;
        Kokkos::parallel_reduce("update_sets", graph.V, KOKKOS_LAMBDA(int u, bool &local_changes) {
            if (inMIS(u) == 1) {
                removed(u) = 1;
                for (int v = 0; v < graph.V; ++v) {
                    if (graph.adj(u, v) == 1) {
                        removed(v) = 1;
                    }
                }
                local_changes = true; // If any vertex is added, flag a change
            }
        }, changes);

    } while (changes);

    return inMIS;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        //Initialize graph
        int V = 6;
        Kokkos::View<int**> adj ("adj",V,V);
        auto h_graph = Kokkos::create_mirror_view(graph.adj);
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
        std::cout << "Maximum Independent Set (MIS) nodes:" << std::endl;
        Kokkos::parallel_for("print_results", V, KOKKOS_LAMBDA(int i) {
            if (independentSet(i) == 1) {
                printf("%d ", i);
            }
        });
        std::cout << std::endl;
    }

    Kokkos::finalize();
    return 0;
}
