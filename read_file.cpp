#include <Kokkos_Core.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

void readGraphFromFile(const std::string &filename, Kokkos::View<int*> xadj, Kokkos::View<int*> adjncy) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open graph file.");
    }

    // Step 1: Read edges and find the max vertex ID
    std::vector<std::pair<int, int>> edges;
    int maxVertex = -1;
    std::string line;

    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) {
            continue; // Skip malformed lines
        }
        edges.emplace_back(u, v);
        edges.emplace_back(v, u); 
        maxVertex = std::max({maxVertex, u, v});
    }

    inputFile.close();

    int numVertices = maxVertex + 1;
    std::vector<int> degree(numVertices, 0);

    // Step 2: Count the degree of each vertex
    for (const auto &edge : edges) {
        degree[edge.first]++;
    }

    // Step 3: Build xadj based on degree information
    std::vector<int> h_xadj(numVertices + 1, 0);
    for (int i = 1; i <= numVertices; ++i) {
        h_xadj[i] = h_xadj[i - 1] + degree[i - 1];
    }

    // Step 4: Build adjncy by filling neighbors
    std::vector<int> h_adjncy(edges.size());
    std::vector<int> currentOffset = h_xadj;

    for (const auto &edge : edges) {
        int u = edge.first;
        int v = edge.second;
        h_adjncy[currentOffset[u]++] = v;
    }

    // Step 5: Create Kokkos views and copy data to device
    xadj = Kokkos::View<int*>("xadj", numVertices + 1);
    adjncy = Kokkos::View<int*>("adjncy", edges.size());

    auto h_xadj_k = Kokkos::create_mirror_view(xadj);
    auto h_adjncy_k = Kokkos::create_mirror_view(adjncy);

    for (size_t i = 0; i < h_xadj.size(); ++i) {
        h_xadj_k(i) = h_xadj[i];
    }

    for (size_t i = 0; i < h_adjncy.size(); ++i) {
        h_adjncy_k(i) = h_adjncy[i];
    }

    Kokkos::deep_copy(xadj, h_xadj_k);
    Kokkos::deep_copy(adjncy, h_adjncy_k);
}
