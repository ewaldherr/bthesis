#include <Kokkos_Core.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <stdexcept>
#include <algorithm>
#include <iostream>

void readGraphFromFile(const std::string &filename, Kokkos::View<int*>& xadj, Kokkos::View<int*>& adjncy) {
    std::cout << "Reading METIS file " << filename << std::endl;

    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Failed to open METIS graph file.");
    }

    // Read the header
    int numVertices, numEdges;
    inputFile >> numVertices >> numEdges;
    if (numVertices <= 0 || numEdges <= 0) {
        throw std::runtime_error("Invalid header in METIS graph file.");
    }

    // Resize xadj and temporary storage for adjacency list
    std::vector<int> v_xadj(numVertices + 1, 0);
    std::vector<int> edges;

    // Read adjacency lists
    std::string line;
    std::getline(inputFile, line); // Skip to the next line after header
    int currentIndex = 0;

    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int neighbor;

        while (iss >> neighbor) {
            edges.push_back(neighbor - 1); // Convert 1-based to 0-based indexing
        }

        v_xadj[currentIndex + 1] = edges.size();
        currentIndex++;
    }

    if (currentIndex != numVertices) {
        throw std::runtime_error("Mismatch between declared and actual number of vertices.");
    }

    inputFile.close();

    // Resize Kokkos Views
    Kokkos::resize(xadj, numVertices + 1);
    Kokkos::resize(adjncy, edges.size());

    auto h_xadj = Kokkos::create_mirror_view(xadj);
    auto h_adjncy = Kokkos::create_mirror_view(adjncy);

    // Copy data into host views
    std::copy(v_xadj.begin(), v_xadj.end(), h_xadj.data());
    std::copy(edges.begin(), edges.end(), h_adjncy.data());

    // Transfer data to device
    Kokkos::deep_copy(xadj, h_xadj);
    Kokkos::deep_copy(adjncy, h_adjncy);

    std::cout << "METIS graph loaded successfully!" << std::endl;
}


