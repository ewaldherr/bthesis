#include <Kokkos_Core.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <iostream>

void readGraphFromFile(const std::string &filename, Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy) {
    std::cout << "Reading in " << filename << std::endl;

    // Open the file
    std::ifstream inputFile(filename, std::ios::in);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open graph file.");
    }

    // Determine file size for chunking
    inputFile.seekg(0, std::ios::end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    // Preallocate memory for edge counting
    int maxVertex = -1;
    std::vector<std::pair<int, int>> edges;
    edges.reserve(fileSize / 16); // Estimate based on average edge size

    // Read edges and determine max vertex ID
    std::string line;
    while (std::getline(inputFile, line)) {
        int u, v;
        std::istringstream iss(line);
        if (!(iss >> u >> v)) continue;

        edges.emplace_back(min(u, v),max(u,v));
        maxVertex = std::max({maxVertex, u, v});
    }
    inputFile.close();

    int numVertices = maxVertex + 1;

    // Build degree information
    std::vector<int> degree(numVertices, 0);
    for (const auto &edge : edges) {
        degree[edge.first]++;
        degree[edge.second]++;
    }

    // Build xadj and adjncy
    std::vector<int> v_xadj(numVertices + 1, 0);
    for (int i = 1; i <= numVertices; ++i) {
        v_xadj[i] = v_xadj[i - 1] + degree[i - 1];
    }

    std::vector<int> v_adjncy(v_xadj.back());
    std::vector<int> offsets = v_xadj;
    for (const auto &edge : edges) {
        v_adjncy[offsets[edge.first]++] = edge.second;
        v_adjncy[offsets[edge.second]++] = edge.first;
    }

    // Resize and copy to Kokkos views
    Kokkos::resize(xadj, numVertices + 1);
    Kokkos::resize(adjncy, edges.size());

    auto h_xadj = Kokkos::create_mirror_view(xadj);
    auto h_adjncy = Kokkos::create_mirror_view(adjncy);

    std::copy(v_xadj.begin(), v_xadj.end(), h_xadj.data());
    std::copy(v_adjncy.begin(), v_adjncy.end(), h_adjncy.data());

    Kokkos::deep_copy(xadj, h_xadj);
    Kokkos::deep_copy(adjncy, h_adjncy);

    std::cout << "Graph loaded successfully." << std::endl;
}

