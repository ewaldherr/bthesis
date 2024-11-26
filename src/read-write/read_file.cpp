#include <Kokkos_Core.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <iostream>

void readGraphFromFile(const std::string &filename, Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy) {
    std::cout << "Reading " << filename << std::endl;

    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open graph file.");
    }

    // First Pass: Find max vertex ID and count degrees
    int maxVertex = -1;
    int u, v;
    std::vector<int> degree;

    while (inputFile >> u >> v) {
        if (u < 0 || v < 0 || u == v) continue; // Skip invalid edges
        maxVertex = std::max({maxVertex, u, v});
        if (maxVertex >= degree.size()) degree.resize(maxVertex + 1, 0);
        degree[u]++;
        degree[v]++;
    }

    int numVertices = maxVertex + 1;

    // Build xadj
    std::vector<int> v_xadj(numVertices + 1, 0);
    for (int i = 1; i <= maxVertex; ++i) {
        v_xadj[i] = v_xadj[i - 1] + degree[i - 1];
    }

    // Second Pass: Populate adjncy
    inputFile.clear();
    inputFile.seekg(0, std::ios::beg);

    std::vector<int> v_adjncy(v_xadj.back());
    std::vector<int> currentOffset = v_xadj;

    while (inputFile >> u >> v) {
        if (u < 0 || v < 0 || u == v) continue; // Skip invalid edges
        v_adjncy[currentOffset[u]++] = v;
        v_adjncy[currentOffset[v]++] = u;
    }

    inputFile.close();

    // Convert to Kokkos Views
    Kokkos::resize(xadj, numVertices + 1);
    Kokkos::resize(adjncy, v_adjncy.size());

    auto h_xadj = Kokkos::create_mirror_view(xadj);
    auto h_adjncy = Kokkos::create_mirror_view(adjncy);

    std::copy(v_xadj.begin(), v_xadj.end(), h_xadj.data());
    std::copy(v_adjncy.begin(), v_adjncy.end(), h_adjncy.data());

    Kokkos::deep_copy(xadj, h_xadj);
    Kokkos::deep_copy(adjncy, h_adjncy);

    std::cout << "Graph loaded successfully!" << std::endl;
}

