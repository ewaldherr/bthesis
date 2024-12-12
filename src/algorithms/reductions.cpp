#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <string>

KOKKOS_FUNCTION void includeTrivial(Kokkos::View<int*>& degree, Kokkos::View<int*>& state){
    Kokkos::parallel_for("include_trivial", degree.extent(0), KOKKOS_LAMBDA(int i) {
        if (degree(i) == 1) {
            state(i) = 1;
            state(adjncy(xadj(i))) = 0;
        }
    });
}

KOKKOS_FUNCTION void includeTriangle(Kokkos::View<int*>& degree, Kokkos::View<int*>& state){
    Kokkos::parallel_for("include_triangle", degree.extent(0), KOKKOS_LAMBDA(int i) {
        if (degree(i) == 2) {
            int v = adjncy(xadj(i));
            int w = adjncy(xadj(i)+1);
            if(degree(v) == 2 || degree(w) == 2){
                return;
            }
            bool triangular = false;
            for(int u = xadj(v); u < xadj(v+1): ++u){
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

KOKKOS_FUNCTION void lowDegree(Kokkos::View<int*>& degree, Kokkos::View<int*>& state){
    Kokkos::parallel_for("low_degree", degree.extent(0), KOKKOS_LAMBDA(int i) {
        if (degree(i) == 1) {
            state(i) = 1;
            state(adjncy(xadj(i))) = 0;
        }
        if (degree(i) == 2) {
            int v = adjncy(xadj(i));
            int w = adjncy(xadj(i)+1);
            if(degree(v) == 2 || degree(w) == 2){
                return;
            }
            bool triangular = false;
            for(int u = xadj(v); u < xadj(v+1): ++u){
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