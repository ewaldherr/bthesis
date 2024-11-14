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
        Kokkos::deep_copy(best_solution,current_solution);
    }
}

KOKKOS_FUNCTION void removeAtRandom(Kokkos::View<int*>& xadj , Kokkos::View<int*>& adjncy, Kokkos::View<int*>& current_solution, double probability){
    Kokkos::Random_XorShift64_Pool<> random_pool((unsigned int)time(NULL));
    Kokkos::parallel_for("remove_vertices", current_solution.extent(0), KOKKOS_LAMBDA(int i) {
        if(current_solution(i)==0) return;
        auto generator = random_pool.get_state();
        if(generator.drand(0.,1.)<= probability){
            current_solution(i) = -1;
            for (int v = xadj(i); v < xadj(i+1); ++v) {
                current_solution(adjncy(v)) = -1;
            }
        }
        random_pool.free_state(generator);
    });
}

// Iterative Luby's Algorithm with removing vertices from the solution 
Kokkos::View<int*> LubyIterAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, int iterations) {
    int size = 0;
    int& best_size = size;
    Kokkos::View<int*> current_solution("current_solution", xadj.extent(0)-1);
    Kokkos::View<int*> best_solution("best_solution", xadj.extent(0)-1);
    auto h_current = Kokkos::create_mirror_view(current_solution);
    Kokkos::deep_copy(current_solution, -1);
    Kokkos::deep_copy(best_solution, -1);

    for(int i =0; i < iterations; ++i){
        current_solution = lubysAlgorithm(xadj, adjncy, current_solution);
        checkSize(best_solution, current_solution, best_size);
        removeAtRandom(xadj, adjncy, current_solution, 0.5);
    }

    return best_solution;
}