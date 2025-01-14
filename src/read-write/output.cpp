#include <Kokkos_Core.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem> // C++17 or later

void writeIndependentSetToFile(const Kokkos::View<int*> independentSet, const std::string &filename) {
    // Ensure the "results" directory exists
    std::filesystem::path dir("results");
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directory(dir);
    }

    // Open the output file in write mode
    std::ofstream outputFile(dir / filename); // Use directory path with filename
    if (!outputFile.is_open()) {
        throw std::runtime_error("Failed to open output file.");
    }
    
    auto h_independentSet = Kokkos::create_mirror_view(independentSet);
    Kokkos::deep_copy(h_independentSet, independentSet);

    int size = 0;
    // Loop through the entries in the independent set and write each entry to a new line
    for (size_t i = 0; i < h_independentSet.extent(0); ++i) {
        outputFile << h_independentSet(i) << "\n";
        if (h_independentSet(i) == 1) size++;
    }
    std::cout << "There are " << size << " vertices inside of the MIS" << std::endl;

    // Close the file
    outputFile.close();
    std::cout << "Result saved in " << filename << std::endl;
}
