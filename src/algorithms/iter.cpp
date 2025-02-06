#include "degree_based.cpp"

KOKKOS_FUNCTION int checkSize(Kokkos::View<int*>& best_solution, Kokkos::View<int*>& current_solution, int& best_size, bool& newBest){
    int size = 0;
    Kokkos::parallel_reduce ("Reduction", current_solution.extent(0), KOKKOS_LAMBDA (const int i, int& sum) {
        if (current_solution(i) == 1) sum++;
    }, size);
    if(size > best_size){
        newBest = true;
        best_size = size;
        Kokkos::deep_copy(best_solution,current_solution);
    }
    return size;
}

KOKKOS_FUNCTION void removeAtRandom(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& current_solution, double probability, unsigned int seed){
    Kokkos::parallel_for("remove_vertices", current_solution.extent(0), KOKKOS_LAMBDA(int i) {
        if(current_solution(i)!=1) return;

        Kokkos::Random_XorShift64<Kokkos::DefaultExecutionSpace> generator(seed + i);
        if(generator.drand(0.,1.)<= probability){
            current_solution(i) = -1;
            for (int v = xadj(i); v < xadj(i+1); ++v) {
                current_solution(adjncy(v)) = -1;
            }
        }
    });
}

KOKKOS_FUNCTION void ensureIndependency(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& current_solution){
    Kokkos::parallel_for("ensure_independency", current_solution.extent(0), KOKKOS_LAMBDA(int i) {
        if(current_solution(i)!=1) return;
        for (int v = xadj(i); v < xadj(i+1); ++v) {
            current_solution(adjncy(v)) = 0;
        }
    });
}

// Iterative Algorithm with removing vertices from the solution 
Kokkos::View<int*> iterAlgorithm(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<int*>& degree, std::string algorithm, unsigned int seed, int time) {
    auto algo_start = std::chrono::high_resolution_clock::now();
    auto algo_stop = std::chrono::high_resolution_clock::now();
    auto algo_duration = std::chrono::duration_cast<std::chrono::microseconds>(algo_stop - algo_start);
    int size = 0;
    int& best_size = size;
    bool best = false;
    bool& newBest = best;
    Kokkos::View<int*> current_solution("current_solution", xadj.extent(0)-1);
    Kokkos::View<int*> best_solution("best_solution", xadj.extent(0)-1);
    Kokkos::deep_copy(current_solution, -1);
    Kokkos::deep_copy(best_solution, -1);
    int totalIterations = 0;

    algorithm = "DEGREEUD";
    while(algo_duration.count()/1000000 < time){
        current_solution = degreeBasedAlgorithm(xadj, adjncy, degree, current_solution, seed + 1000 * totalIterations, algorithm, 1);
        int newSize = checkSize(best_solution, current_solution, best_size, newBest);
        if(newBest){
            newBest = false;
            algo_stop = std::chrono::high_resolution_clock::now();
            algo_duration = std::chrono::duration_cast<std::chrono::microseconds>(algo_stop - algo_start);
            double time = algo_duration.count();
            std::cout << "New best solution found of size " << newSize << " [" << time/1000 << "]" << std::endl;
        }
        //Kokkos::deep_copy(current_solution, best_solution);
        removeAtRandom(xadj, adjncy, current_solution, 0.5, seed + 1000 * totalIterations);
        ensureIndependency(xadj,adjncy,current_solution);
        //updateDegrees(xadj, adjncy, current_solution, degree);
        ++totalIterations;
        algo_stop = std::chrono::high_resolution_clock::now();
        algo_duration = std::chrono::duration_cast<std::chrono::microseconds>(algo_stop - algo_start);
    }

    std::cout << "Iterative approach lasted a total of " << totalIterations << " iterations." << std::endl;
    std::cout << "The found solution has size " << size << std::endl;
    return best_solution;
}