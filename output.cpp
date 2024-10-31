#include <Kokkos_Core.hpp>
#include <fstream>
#include <string>

void writeIndependentSetToFile(const Kokkos::View<int*> independentSet, const std::string &filename) {
    // Open the output file in write mode
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        throw std::runtime_error("Failed to open output file.");
    }
    
    // Loop through the entries in the Kokkos::View and write each entry to a new line
    auto hostindependentSet = Kokkos::create_mirror_view(independentSet);
    Kokkos::deep_copy(hostindependentSet, independentSet);

    for (size_t i = 0; i < hostindependentSet.extent(0); ++i) {
        outputFile << hostindependentSet(i) << "\n";
    }
    
    // Close the file
    outputFile.close();
}
