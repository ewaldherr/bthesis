#include "degree_based.cpp"

KOKKOS_FUNCTION void updateDegrees(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& current_solution, Kokkos::View<int*> degree){
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

//TODO: parallize with Kokkos parallel_reduce
KOKKOS_FUNCTION void checkSize(Kokkos::View<int*>& best_solution, Kokkos::View<int*>& current_solution, Kokkos::View<int*>& best_size){
    int size = 0;
    Kokkos::parallel_reduce ("Reduction", N, KOKKOS_LAMBDA (const int i, int& sum) {
        if (h_current(i) == 1) sum++;
    }, size);
    if(size > best_size(0)){
        best_size(0) = size;
        Kokkos::deep_copy(best_solution,current_solution);
    }
}

KOKKOS_FUNCTION void removeAtRandom(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& current_solution, double probability){
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

// Iterative Algorithm with removing vertices from the solution 
Kokkos::View<int*> iterAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, int iterations, Kokkos::View<int*> degree, std::string algorithm, unsigned int seed) {
    Kokkos::View<int*> best_size ("best_size", 1);
    Kokkos::deep_copy(best_size, 0);
    Kokkos::View<int*> current_solution("current_solution", xadj.extent(0)-1);
    Kokkos::View<int*> best_solution("best_solution", xadj.extent(0)-1);
    auto h_current = Kokkos::create_mirror_view(current_solution);
    Kokkos::deep_copy(current_solution, -1);
    Kokkos::deep_copy(best_solution, -1);

    if(algorithm.compare("LUBYITER") == 0){
        for(int i =0; i < iterations; ++i){
            current_solution = lubysAlgorithm(xadj, adjncy, current_solution, seed + i);
            checkSize(best_solution, current_solution, best_size);
            removeAtRandom(xadj, adjncy, current_solution, 0.5);
        }
    } else{
        for(int i =0; i < iterations; ++i){
            current_solution = degreeBasedAlgorithm(xadj, adjncy, degree, current_solution, seed + i);
            checkSize(best_solution, current_solution, best_size);
            if (i != iterations -1){
                removeAtRandom(xadj, adjncy, current_solution, 0.5);
                updateDegrees(xadj, adjncy, current_solution, degree);
            }
        }
    }

    return best_solution;
}