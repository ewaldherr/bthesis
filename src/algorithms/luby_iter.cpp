#include "degree_based.cpp"

void checkSize(Kokkos::View<int*>& best_solution, Kokkos::View<int*>& current_solution){

}

KOKKOS_FUNCTION void removeAtRandom(Kokkos::View<int*>& current_solution){
    
}

// Degree-based version of Luby's Algorithm
Kokkos::View<int*> LubyIterAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, int iterations) {
    Kokkos::View<int*> current_solution("current_solution", xadj.extent(0)-1);
    Kokkos::View<int*> best_solution("best_solution", xadj.extent(0)-1);
    auto h_current = Kokkos::create_mirror_view(current_solution);
    Kokkos::deep_copy(state, -1);

    for(int i =0; i < iterations; ++i){
        current_solution = lubysAlgorithm(xadj,adjncy);
        checkSize(best_solution, current_solution);
        removeAtRandom(current_solution);
    }

    return state;
}