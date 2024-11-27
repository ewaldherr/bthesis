#include "luby.cpp"

//Function that updates the degrees of each vertex
KOKKOS_FUNCTION void updateDegrees(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& current_solution, Kokkos::View<int*>& degree){
    Kokkos::parallel_for("update_degrees", degree.extent(0), KOKKOS_LAMBDA(int i) {
        if(current_solution(i) != -1) return;
        degree(i) = 0;
        for (int v = xadj(i); v < xadj(i+1); ++v) {
            if(current_solution(adjncy(v)) == -1){
                degree(i)++;
            }
        }
    });
}

// Function that checks for each vertex if it has the max priority of its neighborhood
KOKKOS_FUNCTION void checkMaxDegreePrio(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& degree, Kokkos::View<double*>& priorities, Kokkos::View<int*>& state){
    Kokkos::parallel_for("select_max_priority", xadj.extent(0)-1, KOKKOS_LAMBDA(int u) {
            if (state(u) != -1) return;

            bool isMaxPriority = true;
            for (int v = xadj(u); v < xadj(u+1); ++v) {
                bool isSmaller = true;
                if(degree(u) < degree(adjncy(v))) isSmaller = false;
                if(degree(u) == degree(adjncy(v))){
                    if(priorities(u) > priorities(adjncy(v))) isSmaller = false;
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
Kokkos::View<int*> degreeBasedAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, Kokkos::View<int*> degree,  Kokkos::View<int*>& state, 
    unsigned int seed, std::string algorithm, int updateFrequency) {
    Kokkos::View<double*> priorities("priorities", xadj.extent(0)-1);

    auto h_state = Kokkos::create_mirror_view(state);
    Kokkos::deep_copy(state, -1);

    // Assign random priorities to remaining vertices
    initializePriorities(priorities, seed);
    int iter = 0;
    bool changes;
    do {
        if(iter == updateFrequency && algorithm.compare("DEGREEUD") == 0){
            updateDegrees(xadj, adjncy, state, degree);
            iter = 0;
        }

        // Select vertices with highest priority in their neighborhood
        checkMaxDegreePrio(xadj,adjncy, degree, priorities, state);

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
        ++iter;
    } while (changes);

    return state;
}