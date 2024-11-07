#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <string>
#include <ctime>
#include <chrono>
#include "../read-write/output.cpp"
#include "../read-write/read_file.cpp"
#include "verify_result.cpp"


// Function to initialize random priorities on the GPU
KOKKOS_FUNCTION void initializePriorities(Kokkos::View<double*> priorities) {
    Kokkos::Random_XorShift64_Pool<> random_pool((unsigned int)time(NULL));
    Kokkos::parallel_for("init_priorities", priorities.extent(0), KOKKOS_LAMBDA(int i) {
        auto generator = random_pool.get_state();
        priorities(i) = generator.drand(0.,1.);
        random_pool.free_state(generator);
    });
}

// Function that checks for each vertex if it has the max priority of its neighborhood
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

// Function to remove vertices of vertices added to MIS
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

// Luby's Algorithm
Kokkos::View<int*> lubysAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy) {
    Kokkos::View<int*> state("state", xadj.extent(0)-1);
    Kokkos::View<double*> priorities("priorities", xadj.extent(0)-1);

    auto h_priorities = Kokkos::create_mirror_view(priorities);
    auto h_state = Kokkos::create_mirror_view(state);
    Kokkos::deep_copy(state, -1);

    bool changes;
    do {
        // Assign random priorities to remaining vertices
        initializePriorities(priorities);
        // Select vertices with highest priority in their neighborhood
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
        // Add selected vertices to MIS and remove them and their neighbors
        removeVertices(xadj,adjncy,state);

    } while (changes);

    return state;
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Initialize graph
        Kokkos::View<int*> xadj("xadj",1);
        Kokkos::View<int*> adjncy("adjncy",1);
        // Check if input graph is provided
        if(argc == 1){
            throw std::runtime_error("No input file. Abort program.");
        }
        else{
            readGraphFromFile(argv[1], xadj,adjncy);
            Kokkos::View<int*> result_mis("mis",xadj.extent(0)-1);
            std::cout << "Determining MIS of " << argv[1] << " with " << xadj.extent(0)-1 << " nodes and " << adjncy.extent(0) << " edges."<<std::endl;;
            // Run Luby's algorithm with Kokkos and write results to file
            auto start = std::chrono::high_resolution_clock::now();
            result_mis = lubysAlgorithm(xadj,adjncy);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
            std::cout << "Determined MIS in " << duration.count() << " seconds" << std::endl;

            writeIndependentSetToFile(result_mis,"result_mis.txt");

            if(argc > 2){
                if(strcmp(argv[2],"1") == 0){
                    std::cout << "Verifying solution..." << std::endl;
                    bool valid = verifyResult(result_mis, xadj, adjncy);
                    if(valid){
                        std::cout << "Solution is valid" << std::endl;
                    } else{
                        std::cout << "Solution is NOT valid" << std::endl;
                    }
                }
            }
        }
    }

    Kokkos::finalize();
    return 0;
}
