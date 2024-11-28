#include "degree_based.cpp"

//TODO: parallize with Kokkos parallel_reduce
KOKKOS_FUNCTION int checkSize(Kokkos::View<int*>& best_solution, Kokkos::View<int*>& current_solution, int& best_size){
    int size = 0;
    Kokkos::parallel_reduce ("Reduction", current_solution.extent(0), KOKKOS_LAMBDA (const int i, int& sum) {
        if (current_solution(i) == 1) sum++;
    }, size);
    if(size > best_size){
        best_size = size;
        Kokkos::deep_copy(best_solution,current_solution);
        return best_size;
    }
    return 0;
}

KOKKOS_FUNCTION void removeAtRandom(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& current_solution, double probability, unsigned int seed){
    Kokkos::Random_XorShift64_Pool<> random_pool(seed);
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
Kokkos::View<int*> iterAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, Kokkos::View<int*> degree, std::string algorithm, unsigned int seed) {
    int size = 0;
    int& best_size = size;
    Kokkos::View<int*> current_solution("current_solution", xadj.extent(0)-1);
    Kokkos::View<int*> best_solution("best_solution", xadj.extent(0)-1);
    auto h_current = Kokkos::create_mirror_view(current_solution);
    Kokkos::deep_copy(current_solution, -1);
    Kokkos::deep_copy(best_solution, -1);
    int totalIterations = 0;
    if(algorithm.compare("LUBYITER") == 0){
        for(int i =0; i < 10; ++i){
            current_solution = lubysAlgorithm(xadj, adjncy, current_solution, seed + totalIterations);
            int newBest = checkSize(best_solution, current_solution, best_size);
            if(newBest > 0){
                i = -1;
                std::cout << "New best solution found of size " << newBest << std::endl;
            }
            if(i<9){
                removeAtRandom(xadj, adjncy, current_solution, 0.25, seed + 10 * totalIterations);
            }
            ++totalIterations;
        }
    } else{
        algorithm = "DEGREEUD";
        for(int i =0; i < 10; ++i){
            current_solution = degreeBasedAlgorithm(xadj, adjncy, degree, current_solution, seed + totalIterations, algorithm, 2);
            int newBest = checkSize(best_solution, current_solution, best_size);
            if(newBest > 0){
                i = -1;
                std::cout << "New best solution found of size " << newBest << std::endl;
            }
            if(i<9){
                removeAtRandom(xadj, adjncy, current_solution, 0.25, seed + 10 * totalIterations);
                updateDegrees(xadj, adjncy, current_solution, degree);
            }
            ++totalIterations;
        }
    }
    std::cout << "Iterative approach lasted a total of " << totalIterations << " iterations." << std::endl;
    return best_solution;
}