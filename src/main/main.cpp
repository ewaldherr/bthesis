#include "../read-write/output.cpp"
#include "../read-write/read_file.cpp"
#include "../algorithms/verify_result.cpp"
#include "../algorithms/iter.cpp"

void initializeDegrees(Kokkos::View<int*>& degree, Kokkos::View<int*> xadj){
    auto h_degree = Kokkos::create_mirror_view(degree);
    auto h_xadj = Kokkos::create_mirror_view(xadj);
    for(int i = 0; i < degree.extent(0); ++i){
        h_degree(i) = h_xadj(i+1)-h_xadj(i);
    }
    Kokkos::deep_copy(degree, h_degree);
}

int main(int argc, char* argv[]) {
    auto start = std::chrono::high_resolution_clock::now();
    Kokkos::initialize(argc, argv);
    {
        // Initialize graph
        Kokkos::View<int*> xadj("xadj",1);
        Kokkos::View<int*> adjncy("adjncy",1);
        // Check if input graph is provided
        if(argc == 1){
            throw std::runtime_error("No input file. Abort program.");
        }
        else{
            readGraphFromFile(argv[1], xadj,adjncy);

            // Determining which algorithm to use
            std::string algorithm;
            if(argc == 2){
                std::cout << "No algorithm provided. Continuing with LUBY as default" << std::endl;;
                algorithm = "LUBY";
            } else {
                algorithm = argv[2];
            }
            Kokkos::View<int*> result_mis("mis",xadj.extent(0)-1);
            std::cout << "Determining MIS of " << argv[1] << " with " << xadj.extent(0)-1 << " nodes and " << adjncy.extent(0) << " edges using " << algorithm << "."<< std::endl;;
            // Set up seed for RNG
            unsigned int seed;
            if(argc > 4){
                std::string str_seed= argv[4];
                seed = std::stoi(str_seed);
            } else{
                seed = (unsigned int)time(NULL);
            }
            // Set up degrees
            Kokkos::View<int*> degree("degree", xadj.extent(0)-1);
            initializeDegrees(degree, xadj);
            auto h_degree = Kokkos::create_mirror_view(degree);
            for(int i = 0; i < degree.extent(0); ++i){
                std::cout << h_degree(i) << " ";
            }
            std::cout << std::endl;
            // Run algorithm with Kokkos
            Kokkos::View<int*> state("state", xadj.extent(0)-1);
            auto algo_start = std::chrono::high_resolution_clock::now();
            if(algorithm.compare("DEGREE") == 0){
                result_mis = degreeBasedAlgorithm(xadj, adjncy, degree, state, seed);
            } else if(algorithm.compare("LUBYITER") == 0 || algorithm.compare("DEGREEITER") == 0){
                result_mis = iterAlgorithm(xadj, adjncy, 100, degree, algorithm, seed);
            } else{
                result_mis = lubysAlgorithm(xadj, adjncy, state, seed);
            }
            auto algo_stop = std::chrono::high_resolution_clock::now();
            auto algo_duration = std::chrono::duration_cast<std::chrono::milliseconds>(algo_stop - algo_start);
            std::cout << "Determined MIS in " << algo_duration.count() << " milliseconds" << std::endl;

            // Write results to file
            std::string input_path = argv[1];
            std::string base_filename = input_path.substr(input_path.find_last_of("/\\") + 1);
            writeIndependentSetToFile(result_mis,"mis_" + base_filename);

            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "Main program run for " << duration.count() << " milliseconds" << std::endl;

            if(argc > 3){
                if(strcmp(argv[3],"1") == 0){
                    std::cout << "Verifying solution..." << std::endl;
                    bool valid = verifyResult(result_mis, xadj, adjncy);
                    if(valid){
                        std::cout << "Solution is valid" << std::endl;
                    } else{
                        std::cout << "Solution is NOT valid" << std::endl;
                    }
                }
            }
        }
    }

    Kokkos::finalize();
    return 0;
}