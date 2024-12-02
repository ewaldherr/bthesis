#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <string>

KOKKOS_FUNCTION void includeTrivial(Kokkos::View<int*>& degree, Kokkos::View<int*>& state){
    Kokkos::parallel_for("include_trivial", degree.extent(0), KOKKOS_LAMBDA(int i) {
        if (degree(i) < 2) {
            state(i) = 1;
            for (int v = xadj(i); v < xadj(i+1); ++v) {
                state(adjncy(v)) = 0;
                degree(adjncy(v))--;
            }
        }
    });
}