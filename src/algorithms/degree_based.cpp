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
KOKKOS_FUNCTION int checkMaxDegreePrio(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& degree, Kokkos::View<double*>& priorities, Kokkos::View<int*>& state){
    int changes = 0;
    Kokkos::parallel_reduce("select_max_priority", xadj.extent(0)-1, KOKKOS_LAMBDA(int u, int& vertices) {
            if (state(u) != -1) return;

            bool isMaxPriority = true;
            for (int v = xadj(u); v < xadj(u+1); ++v) {
                bool isSmaller = true;
                if(degree(u) < degree(adjncy(v))) isSmaller = false;
                if(degree(u) == degree(adjncy(v))){
                    if(priorities(u) > priorities(adjncy(v))) isSmaller = false;
                }
                if ((state(adjncy(v)) == -1 && isSmaller) || state(adjncy(v)) == 1) {
                    isMaxPriority = false;
                    break;
                }
            }

            if (isMaxPriority) {
                state(u) = 1;
                for (int v = xadj(u); v < xadj(u+1); ++v) {
                    state(adjncy(v)) = 0;
                    vertices++;
                }
            }
        }, changes);
    return changes;
}

// Degree-based version of Luby's Algorithm
Kokkos::View<int*> degreeBasedAlgorithm(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& degree,  Kokkos::View<int*>& state, 
    unsigned int seed, std::string algorithm, int updateFrequency) {
    Kokkos::View<double*> priorities("priorities", xadj.extent(0)-1);

    auto h_state = Kokkos::create_mirror_view(state);

    // Assign random priorities to remaining vertices
    initializePriorities(priorities, seed);

    //int totalIterations = 0;
    int iter = 1;
    if(updateFrequency == 0){
        ++updateFrequency;
    }
    if(updateFrequency >= 10){
        updateFrequency = 1000000;
    }
    bool changes;
    do {
        if(iter == updateFrequency && algorithm.compare("DEGREEUD") == 0){
            updateDegrees(xadj, adjncy, state, degree);
            iter = 0;
        }
        changes = (checkMaxDegreePrio(xadj,adjncy,degree,priorities,state) > 0);
        ++iter;
        //++totalIterations;
    } while (changes);

    //std::cout << "The algorithm run a total of " << totalIterations << " total iterations" << std::endl;
    return state;
}