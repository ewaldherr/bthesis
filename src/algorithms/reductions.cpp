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

KOKKOS_FUNCTION void includeTriangle(Kokkos::View<int*>& degree, Kokkos::View<int*>& state, Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy){
    Kokkos::parallel_for("include_triangle", degree.extent(0), KOKKOS_LAMBDA(int i) {
        if (degree(i) == 2) {
            int v = adjncy(xadj(i));
            int w = adjncy(xadj(i)+1);
            if(degree(v) == 2 || degree(w) == 2){
                return;
            }
            bool triangular = false;
            for(int u = xadj(v); u < xadj(v+1); ++u){
                if(adjncy(u) == w){
                    triangular = true;
                }
            }
            if(triangular){
                state(i) = 1;
                state(v) = 0;
                state(w) = 0;
            }
        }
    });
}

KOKKOS_FUNCTION void checkDominatedVertices(Kokkos::View<int*>& degree, Kokkos::View<int*>& state, Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy) {
    Kokkos::parallel_for("check_dominated_vertices", degree.extent(0), KOKKOS_LAMBDA(const int i) {
        if (state(i) == 0) return; // Skip already removed vertices

        bool allNeighborsDominate = true; 
        // Iterate through neighbors of vertex i
        for (int j_idx = xadj(i); j_idx < xadj(i + 1); ++j_idx) {
            int j = adjncy(j_idx);  // Neighbor vertex j
            if (state(j) == 0) continue; // Skip already removed vertices

            // Check if j dominates i
            bool jDominatesI = true;
            for (int k_idx = xadj(i); k_idx < xadj(i + 1); ++k_idx) {
                int neighbor_of_i = adjncy(k_idx);
                if (neighbor_of_i == j) continue; // Skip self-check

                // Check if neighbor_of_i is in j's neighborhood
                bool found = false;
                for (int l_idx = xadj(j); l_idx < xadj(j + 1); ++l_idx) {
                    if (adjncy(l_idx) == neighbor_of_i) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    jDominatesI = false; // j does not dominate i
                    break;
                }
            }

            // If j dominates i, mark j for removal
            if (jDominatesI) {
                state(j) = 0; // Neighbor j dominates vertex i
            } else {
                allNeighborsDominate = false; // Not all neighbors dominate i
            }
        }

        // If all neighbors dominate i, mark i as 1 (dominated by all)
        if (allNeighborsDominate) {
            state(i) = 1;
        }
    });
}


KOKKOS_FUNCTION void allReductions(Kokkos::View<int*>& degree, Kokkos::View<int*>& state, Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy){
    Kokkos::parallel_for("low_degree", degree.extent(0), KOKKOS_LAMBDA(int i) {
        // Include trivial vertices
        if (state(i) == 0) return; // Skip already removed vertices

        if (degree(i) == 1 && degree(adjncy(xadj(i))) > 1) {
            state(i) = 1;
            state(adjncy(xadj(i))) = 0;       
        }

        bool allNeighborsDominate = true; 
        // Iterate through neighbors of vertex i
        for (int j_idx = xadj(i); j_idx < xadj(i + 1); ++j_idx) {
            int j = adjncy(j_idx);  // Neighbor vertex j
            if (state(j) == 0) continue; // Skip already removed vertices

            // Check if j dominates i
            bool jDominatesI = true;
            for (int k_idx = xadj(i); k_idx < xadj(i + 1); ++k_idx) {
                int neighbor_of_i = adjncy(k_idx);
                if (neighbor_of_i == j) continue; // Skip self-check

                // Check if neighbor_of_i is in j's neighborhood
                bool found = false;
                for (int l_idx = xadj(j); l_idx < xadj(j + 1); ++l_idx) {
                    if (adjncy(l_idx) == neighbor_of_i) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    jDominatesI = false; // j does not dominate i
                    break;
                }
            }

            // If j dominates i, mark j for removal
            if (jDominatesI) {
                state(j) = 0; // Neighbor j dominates vertex i
            } else {
                allNeighborsDominate = false; // Not all neighbors dominate i
            }
        }

        // If all neighbors dominate i, mark i as 1 (dominated by all)
        if (allNeighborsDominate) {
            state(i) = 1;
        }
    });
}