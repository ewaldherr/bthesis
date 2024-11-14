#include "../src/read-write/read_file.cpp"
#include "../src/algorithms/verify_result.cpp"
#include "../src/algorithms/iter.cpp"

Kokkos::View<int*> initializeDegrees(Kokkos::View<int*> xadj){
    Kokkos::View<int*> degree("degree", xadj.extent(0)-1);
    auto h_degree = Kokkos::create_mirror_view(degree);
    auto h_xadj = Kokkos::create_mirror_view(xadj);
    Kokkos::deep_copy(h_xadj, xadj);
    for(int i = 0; i < degree.extent(0); ++i){
        h_degree(i) = h_xadj(i+1)-h_xadj(i);
    }
    Kokkos::deep_copy(degree, h_degree);
    return degree;
}

int main(int argc, char* argv[]) {
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

            // Set up seed for RNG
            unsigned int seed;
            if(argc > 2){
                std::string str_seed= argv[2];
                seed = std::stoi(str_seed);
            } else{
                seed = (unsigned int)time(NULL);
            }
            // Determining which algorithm to use
            std::string[4] algorithm = {"LUBY","LUBYITER","DEGREE","DEGREEITER"};

            for(auto algo: algorithms){
                Kokkos::View<int*> result_mis("mis",xadj.extent(0)-1);
                std::cout << "Determining MIS of " << argv[1] << " with " << xadj.extent(0)-1 << " nodes and " << adjncy.extent(0) << " edges using " << algo << "."<< std::endl;

                for(int i = 0; i < 10; ++i){
                    // Set up degrees
                    Kokkos::View<int*> degree("degree", xadj.extent(0)-1);
                    degree = initializeDegrees(xadj);
                    
                    // Run algorithm with Kokkos
                    Kokkos::View<int*> state("state", xadj.extent(0)-1);
                    auto algo_start = std::chrono::high_resolution_clock::now();
                    if(algo.compare("DEGREE") == 0){
                        result_mis = degreeBasedAlgorithm(xadj, adjncy, degree, state, seed);
                    } else if(algo.compare("LUBYITER") == 0 || algo.compare("DEGREEITER") == 0){
                        result_mis = iterAlgorithm(xadj, adjncy, 100, degree, algorithm, seed);
                    } else{
                        result_mis = lubysAlgorithm(xadj, adjncy, state, seed);
                    }
                    auto algo_stop = std::chrono::high_resolution_clock::now();
                    auto algo_duration = std::chrono::duration_cast<std::chrono::milliseconds>(algo_stop - algo_start);
                    std::cout << "Determined MIS in " << algo_duration.count() << " milliseconds" << std::endl;

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