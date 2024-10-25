#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>

class Graph {
public:
    int V; // number of vertices
    Kokkos::View<int**> adj; // adjacency matrix

    Graph(int V) : V(V), adj("adj", V, V) {
        // Initialize adjacency matrix to 0
        for(int i = 0; i < V; ++i){
            for (int j = 0; j < V; ++j) {
                adj(i, j) = 0;
            }  
        }  
    }

    void addEdge(int u, int v) {
        adj(u, v) = 1;
        adj(v, u) = 1;
    }
};

// Function to initialize random priorities on the GPU
KOKKOS_FUNCTION Kokkos::View<int*> initializePriorities(int V) {
    Kokkos::View<int*> priorities("priorities", V);
    Kokkos::parallel_for("init_priorities", V, KOKKOS_LAMBDA(int i) {
        unsigned seed = 1234 + i; // Seed for random number generator
        priorities(i) = rand_r(&seed);
    });
    return priorities;
}

// Luby's Algorithm with Kokkos
KOKKOS_FUNCTION Kokkos::View<int*> lubysAlgorithm(Graph &graph) {
    Kokkos::View<int*> inMIS("inMIS", graph.V);
    Kokkos::View<int*> removed("removed", graph.V);
    Kokkos::View<int*> priorities;

    Kokkos::deep_copy(inMIS, 0);
    Kokkos::deep_copy(removed, 0);

    bool changes;
    do {
        // Step 1: Assign random priorities to remaining vertices
        priorities = initializePriorities(graph.V);

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
        // Create a sample graph with 6 nodes
        int V = 6;
        Graph graph(V);
        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(2, 3);
        graph.addEdge(3, 4);
        graph.addEdge(3, 5);

        // Run Luby's algorithm with Kokkos
        Kokkos::View<int*> independentSet = lubysAlgorithm(graph);

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
