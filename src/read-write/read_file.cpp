#include <Kokkos_Core.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <algorithm>
#include <iostream>

void readGraphFromFile(const std::string &filename, Kokkos::View<int*, Kokkos::CudaSpace>& xadj, Kokkos::View<int*, Kokkos::CudaSpace>& adjncy) {
    std::cout << "Reading in " << filename << std::endl;
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open graph file.");
    }

    // Read edges and find the max vertex ID
    std::vector<std::pair<int, int>> edges;
    int maxVertex = -1;
    std::string line;

    // Use a set to track unique edges
    std::set<std::pair<int, int>> edgeSet;

    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int u, v;
        // Skip malformed lines
        if (!(iss >> u >> v)) {
            continue;
        }

        // Ensure no duplicates for undirected graphs by storing both (u, v) and (v, u)
        if (edgeSet.count({u, v}) == 0) {
            edges.emplace_back(u, v);
            edgeSet.insert({u, v});
            maxVertex = std::max({maxVertex, u, v});
        }

        // Ensure backward edges for directed graphs
        if (edgeSet.count({v, u}) == 0) {
            edges.emplace_back(v, u);
            edgeSet.insert({v, u});
            maxVertex = std::max({maxVertex, v, u});
        }
    }

    inputFile.close();

    int numVertices = maxVertex + 1;
    std::vector<int> degree(numVertices, 0);

    // Count the degree of each vertex
    for (const auto &edge : edges) {
        degree[edge.first]++;
    }

    // Build xadj based on degree information
    std::vector<int> v_xadj(numVertices + 1, 0);
    for (int i = 1; i <= numVertices; ++i) {
        v_xadj[i] = v_xadj[i - 1] + degree[i - 1];
    }

    // Build adjncy by filling neighbors
    std::vector<int> v_adjncy(edges.size());
    std::vector<int> currentOffset = v_xadj;

    for (const auto &edge : edges) {
        int u = edge.first;
        int v = edge.second;
        v_adjncy[currentOffset[u]++] = v;
    }

    // Resize Kokkos views
    Kokkos::resize(xadj, numVertices + 1);
    Kokkos::resize(adjncy, edges.size());

    auto h_xadj = Kokkos::create_mirror_view(xadj);
    auto h_adjncy = Kokkos::create_mirror_view(adjncy);

    // Write values to mirror_views
    for (size_t i = 0; i < v_xadj.size(); ++i) {
        h_xadj(i) = v_xadj[i];
    }

    for (size_t i = 0; i < v_adjncy.size(); ++i) {
        h_adjncy(i) = v_adjncy[i];
    }
    // Copy graph information to device
    Kokkos::deep_copy(xadj, h_xadj);
    Kokkos::deep_copy(adjncy, h_adjncy);
}
