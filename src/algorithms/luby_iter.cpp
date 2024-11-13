#include "degree_based.cpp"

void checkSize(Kokkos::View<int*>& best_solution, Kokkos::View<int*>& current_solution, int& best_size){
    auto h_current = Kokkos::create_mirror_view(current_solution);
    Kokkos::deep_copy(h_current, current_solution);

    int size = 0;
    // Loop through the entries in the independent set and write each entry to a new line
    for (size_t i = 0; i < h_current.extent(0); ++i) {
        if (h_current(i) == 1) size++;
    }
    if(size > best_size){
        best_size = size;
        best_solution = current_solution;
    }
}

KOKKOS_FUNCTION void removeAtRandom(Kokkos::View<int*>& current_solution){

}

// Degree-based version of Luby's Algorithm
Kokkos::View<int*> LubyIterAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, int iterations) {
    int size = 0;
    int& best_size = size;
    Kokkos::View<int*> current_solution("current_solution", xadj.extent(0)-1);
    Kokkos::View<int*> best_solution("best_solution", xadj.extent(0)-1);
    auto h_current = Kokkos::create_mirror_view(current_solution);
    Kokkos::deep_copy(state, -1);

    for(int i =0; i < iterations; ++i){
        current_solution = lubysAlgorithm(xadj, adjncy, current_solution);
        checkSize(best_solution, current_solution, best_size);
        removeAtRandom(current_solution);
    }

    return state;
}