#include <Kokkos_Core.hpp>
#include <fstream>
#include <string>

void writeIndependentSetToFile(const Kokkos::View<int*> &independentSetView, const std::string &filename) {
    // Open the output file in write mode
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        throw std::runtime_error("Failed to open output file.");
    }
    
    // Loop through the entries in the Kokkos::View and write each entry to a new line
    auto hostIndependentSetView = Kokkos::create_mirror_view(independentSetView);
    Kokkos::deep_copy(hostIndependentSetView, independentSetView);

    for (size_t i = 0; i < hostIndependentSetView.extent(0); ++i) {
        outputFile << hostIndependentSetView(i) << "\n";
    }
    
    // Close the file
    outputFile.close();
}
