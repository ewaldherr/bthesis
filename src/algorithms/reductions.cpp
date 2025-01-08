#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <string>
#include <ctime>
#include <chrono>

KOKKOS_FUNCTION int countAffected(Kokkos::View<int*>& state){
    int sum = 0;
    Kokkos::parallel_reduce("count_affected", state.extent(0), KOKKOS_LAMBDA(const int i, int& vertices) {
        if(state(i) != -1){
            vertices++;
        }
    }, sum);
    return sum;
}

KOKKOS_FUNCTION void includeTrivial(Kokkos::View<int*>& degree, Kokkos::View<int*>& state, Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy){
    Kokkos::parallel_for("include_trivial", degree.extent(0), KOKKOS_LAMBDA(int i) {
        if (degree(i) == 1  && degree(adjncy(xadj(i))) > 1) {
            state(i) = 1;
            state(adjncy(xadj(i))) = 0;
        }
    });
}

KOKKOS_FUNCTION void includeIsolated(Kokkos::View<int*>& degree, Kokkos::View<int*>& state, Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy) {
    Kokkos::parallel_for("find_isolated_clique_vertices", degree.extent(0), KOKKOS_LAMBDA(int i) {
        if (degree(i) < 2) {
            return; 
        }
        for (int j = xadj(i); j < xadj(i + 1); ++j) {
            int neighbor = adjncy(j);
            // Abort search early if degree is not fitting
            if(degree(neighbor) < degree(i)){
                return;
            }
            if(degree(neighbor) == degree(i)){
                if(neighbor < i){
                    return;
                }
            }
            // Check if `neighbor` is connected to all other neighbors of `i`
            for (int k = j + 1; k < xadj(i + 1); ++k) {
                int other_neighbor = adjncy(k);
                if (neighbor == other_neighbor) {
                    continue;
                }

                // Check if `neighbor` is connected to `other_neighbor`
                bool connected = false;
                for (int l = xadj(neighbor); l < xadj(neighbor + 1); ++l) {
                    if (adjncy(l) == other_neighbor) {
                        connected = true;
                        break;
                    }
                }
                if (!connected) {
                    return;
                }
            }
        }
        state(i) = 1;
        for(int v = xadj(i); v < xadj(i+1); ++v){
            state(adjncy(v)) = 0;
        }
    });
}

KOKKOS_FUNCTION void allRed(Kokkos::View<int*>& degree, Kokkos::View<int*>& state, Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy){
    Kokkos::parallel_for("low_degree", degree.extent(0), KOKKOS_LAMBDA(int i) {
        // Include trivial vertices
        if (degree(i) == 1 && degree(adjncy(xadj(i))) > 1) {
            state(i) = 1;
            state(adjncy(xadj(i))) = 0;       
        }
        else{
            // Include isolated vertices
            for (int j = xadj(i); j < xadj(i + 1); ++j) {
                int neighbor = adjncy(j);
                // Abort search early if degree is not fitting
                if(degree(neighbor) < degree(i)){
                    return;
                }
                if(degree(neighbor) == degree(i)){
                    if(neighbor < i){
                        return;
                    }
                }
                // Check if neighbor is connected to all other neighbors of `i`
                for (int k = j + 1; k < xadj(i + 1); ++k) {
                    int other_neighbor = adjncy(k);

                    // Check if neighbor is connected to other_neighbor
                    bool connected = false;
                    for (int l = xadj(neighbor); l < xadj(neighbor + 1); ++l) {
                        if (adjncy(l) == other_neighbor) {
                            connected = true;
                            break;
                        }
                    }
                    if (!connected) {
                        return;
                    }
                }
            }
            state(i) = 1;
            for(int v = xadj(i); v < xadj(i+1); ++v){
                state(adjncy(v)) = 0;
            }
        }
    });
}