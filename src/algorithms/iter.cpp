#include "degree_based.cpp"

KOKKOS_FUNCTION int checkSize(Kokkos::View<int*>& best_solution, Kokkos::View<int*>& current_solution, int& best_size){
    int size = 0;
    Kokkos::parallel_reduce ("Reduction", current_solution.extent(0), KOKKOS_LAMBDA (const int i, int& sum) {
        if (current_solution(i) == 1) sum++;
    }, size);
    if(size > best_size){
        best_size = size;
        Kokkos::deep_copy(best_solution,current_solution);
    }
    return size;
}

KOKKOS_FUNCTION void removeAtRandom(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& current_solution, double probability, unsigned int seed){
    Kokkos::parallel_for("remove_vertices", current_solution.extent(0), KOKKOS_LAMBDA(int i) {
        if(current_solution(i)==0) return;

        Kokkos::Random_XorShift64<Kokkos::DefaultExecutionSpace> generator(seed + i);
        if(generator.drand(0.,1.)<= probability){
            current_solution(i) = -1;
            for (int v = xadj(i); v < xadj(i+1); ++v) {
                current_solution(adjncy(v)) = -1;
            }
        }
    });
}

// Iterative Algorithm with removing vertices from the solution 
Kokkos::View<int*> iterAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, Kokkos::View<int*> degree, std::string algorithm, unsigned int seed) {
    auto algo_start = std::chrono::high_resolution_clock::now();
    auto algo_stop = std::chrono::high_resolution_clock::now();
    auto algo_duration = std::chrono::duration_cast<std::chrono::seconds>(algo_stop - algo_start);
    int size = 0;
    int& best_size = size;
    Kokkos::View<int*> current_solution("current_solution", xadj.extent(0)-1);
    Kokkos::View<int*> best_solution("best_solution", xadj.extent(0)-1);
    auto h_current = Kokkos::create_mirror_view(current_solution);
    Kokkos::deep_copy(current_solution, -1);
    Kokkos::deep_copy(best_solution, -1);
    int totalIterations = 0;

    algorithm = "DEGREEUD";
    while(algo_duration.count() < 120){
        current_solution = degreeBasedAlgorithm(xadj, adjncy, degree, current_solution, seed + totalIterations, algorithm, 1);
        int newSize = checkSize(best_solution, current_solution, best_size);
        if(newSize > size){
            algo_stop = std::chrono::high_resolution_clock::now();
            algo_duration = std::chrono::duration_cast<std::chrono::seconds>(algo_stop - algo_start);
            std::cout << "New best solution found of size " << newBest << " [" << algo_duration.count() << "]" << std::endl;
        }
        removeAtRandom(xadj, adjncy, current_solution, 0.5, seed + 1000 * totalIterations);
        //updateDegrees(xadj, adjncy, current_solution, degree);
        ++totalIterations;
        std::cout << "Size found is " << size << std::endl;
        algo_stop = std::chrono::high_resolution_clock::now();
        algo_duration = std::chrono::duration_cast<std::chrono::seconds>(algo_stop - algo_start);
    }

    std::cout << "Iterative approach lasted a total of " << totalIterations << " iterations." << std::endl;
    return best_solution;
}