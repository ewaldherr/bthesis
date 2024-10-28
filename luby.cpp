#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>

// Function to initialize random priorities on the GPU
KOKKOS_FUNCTION void initializePriorities(Kokkos::View<double*> priorities) {
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    Kokkos::parallel_for("init_priorities", priorities.extent(0), KOKKOS_LAMBDA(int i) {
        auto generator = random_pool.get_state();
        priorities(i) = generator.drand(0., 1.);
    });
}

KOKKOS_FUNCTION void checkMax(Kokkos::View<int**> graph,Kokkos::View<double*> priorities,Kokkos::View<int*> state){
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
}

KOKKOS_FUNCTION void removeVertices(Kokkos::View<int**> graph, Kokkos::View<int*> state){
    Kokkos::parallel_for("update_sets", graph.extent(1), KOKKOS_LAMBDA(int u) {
        if (state(u) == 1) {
            state(u) = 2;
            for (int v = 0; v < graph.extent(1); ++v) {
                if (graph(u, v) == 1) {
                    state(v) = -1;
                }
            }
        }
    });
}

// Luby's Algorithm with Kokkos
Kokkos::View<int*> lubysAlgorithm(Kokkos::View<int**> graph) {
    Kokkos::View<int*> state("state", graph.extent(1));
    Kokkos::View<double*> priorities("priorities", graph.extent(1));
    auto h_state = Kokkos::create_mirror_view(state);
    auto h_priorities = Kokkos::create_mirror_view(priorities);
    Kokkos::deep_copy(state, 0);
    int iter = 0;
    bool changes;
    do {
        // Step 1: Assign random priorities to remaining vertices
        initializePriorities(priorities);
        Kokkos::deep_copy(h_priorities,priorities);
        Kokkos::deep_copy(h_state,state);
        for(int i = 0; i < state.extent(0);++i){
            std::cout << h_priorities(i) << " " << h_state(i) << " ";
        }
        std::cout << std::endl;
        // Step 2: Select vertices with highest priority in their neighborhood
        checkMax(graph,priorities,state);
        Kokkos::deep_copy(h_priorities,priorities);
        Kokkos::deep_copy(h_state,state);
        for(int i = 0; i < state.extent(0);++i){
            std::cout << h_priorities(i) << " " << h_state(i) << " ";
        }
        std::cout << std::endl;
        // Step 3: Add selected vertices to MIS and remove them and their neighbors
        removeVertices(graph,state);
        Kokkos::deep_copy(h_priorities,priorities);
        Kokkos::deep_copy(h_state,state);
        for(int i = 0; i < state.extent(0);++i){
            std::cout << h_priorities(i) << " " << h_state(i) << " ";
        }
        for(int i = 0; i < state.extent(0);++i){
            if(h_state(i)==1){
                changes = true;
                break;
            }
        }
        std::cout << std::endl;
        ++iter;
    } while (changes);
    std::cout << iter << std::endl;
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
