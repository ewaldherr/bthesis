#ifndef LUBY_HPP
#define LUBY_HPP

// Declarations only if this is a header
KOKKOS_FUNCTION void initializePriorities(Kokkos::View<double*> priorities);
KOKKOS_FUNCTION void checkMax(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, Kokkos::View<double*> priorities, Kokkos::View<int*> state);
KOKKOS_FUNCTION void removeVertices(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy, Kokkos::View<int*> state);
Kokkos::View<int*> lubysAlgorithm(Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy);

#endif // LUBY_HPP
