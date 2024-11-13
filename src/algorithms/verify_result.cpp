#include <Kokkos_Core.hpp>
#include <iostream>

// Check for each vertex inside of solution if another vertex in solution is adjacent
void checkVertices(Kokkos::View<int*> result_mis, Kokkos::View<bool*> valid, Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy){
    Kokkos::parallel_for("check_vertices", valid.extent(0), KOKKOS_LAMBDA(int u) {
        valid(u) = true;
        // Check for maximality
        if(result_mis(u) == 0){
            for (int v = xadj(u); v < xadj(u+1); ++v) {
                if (result_mis(adjncy(v)) == 1) {
                    return;
                }
            }
            valid(u) = false;
        // Check for independency
        } else {
            for (int v = xadj(u); v < xadj(u+1); ++v) {
                if (result_mis(adjncy(v)) == 1) {
                    valid(u) = false;
                    break;
                }
            }
        }
    });
}

bool verifyResult(Kokkos::View<int*> result_mis, Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy){
    Kokkos::View<bool*> valid ("valid", result_mis.extent(0));
    auto h_valid = Kokkos::create_mirror_view(valid);

    checkVertices(result_mis, valid, xadj, adjncy);
    Kokkos::deep_copy(h_valid,valid);

    for(int i = 0; i < valid.extent(0); ++i){
        if(!h_valid(i)) return false;
    }
    return true;
}

