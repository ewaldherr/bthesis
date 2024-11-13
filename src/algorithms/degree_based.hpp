#ifndef DEGREEBASED_HPP
#define DEGREEBASED_HPP

// Declarations only if this is a header
KOKKOS_FUNCTION void checkMaxDegreePrio(Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy, Kokkos::View<double*>& priorities, Kokkos::View<int*>& state);
Kokkos::View<int*> degreeBasedAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy);

#endif // DEGREEBASED_HPP
