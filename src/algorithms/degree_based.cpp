#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <string>
#include <ctime>
#include <chrono>
#include "luby.cpp"

// Function that checks for each vertex if it has the max priority of its neighborhood
KOKKOS_FUNCTION void checkMaxDegreePrio(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<double*>& priorities, Kokkos::View<int*>& state){
    Kokkos::parallel_for("select_max_priority", xadj.extent(0)-1, KOKKOS_LAMBDA(int u) {
            if (state(u) != -1) return;

            bool isMaxPriority = true;
            for (int v = xadj(u); v < xadj(u+1); ++v) {
                bool isSmaller = true;
                if(xadj(u+1)-xadj(u) < xadj(v+1)-xadj(v)) isSmaller = false;
                if(xadj(u+1)-xadj(u) == xadj(v+1)-xadj(v)){
                    if(priorities(adjncy(u)) > priorities(adjncy(v))) isSmaller = false;
                }
                if ((state(adjncy(v)) == -1 && isSmaller) || state(adjncy(v)) == 2) {
                    isMaxPriority = false;
                    break;
                }
            }

            if (isMaxPriority) {
                state(u) = 2;
            }
        });
}

// Degree-based version of Luby's Algorithm
Kokkos::View<int*> degreeBasedAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy) {
    Kokkos::View<int*> state("state", xadj.extent(0)-1);
    Kokkos::View<double*> priorities("priorities", xadj.extent(0)-1);

    auto h_priorities = Kokkos::create_mirror_view(priorities);
    auto h_state = Kokkos::create_mirror_view(state);
    Kokkos::deep_copy(state, -1);

    bool changes;
    do {
        // Assign random priorities to remaining vertices
        initializePriorities(priorities);
        Kokkos::deep_copy(h_priorities,priorities);
        Kokkos::deep_copy(h_state,state);
        for(int i = 0; i < state.extent(0);++i){
            std::cout << h_state(i) << " " << h_priorities(i) << " "; 
        }
        std::cout << std::endl;
        // Select vertices with highest priority in their neighborhood
        checkMaxDegreePrio(xadj,adjncy,priorities,state);
        Kokkos::deep_copy(h_priorities,priorities);
        Kokkos::deep_copy(h_state,state);
        for(int i = 0; i < state.extent(0);++i){
            std::cout << h_state(i) << " " << h_priorities(i) << " "; 
        }
        std::cout << std::endl;
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