#include "../src/read-write/read_file.cpp"
#include "../src/algorithms/verify_result.cpp"
#include "../src/algorithms/iter.cpp"

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

            // Determining which algorithm to use
            std::string algorithm;
            if(argc == 2){
                std::cout << "No algorithm provided. Continuing with LUBY as default" << std::endl;;
                algorithm = "LUBY";
            } else {
                algorithm = argv[2];
            }
            Kokkos::View<int*> result_mis("mis",xadj.extent(0)-1);
            std::cout << "Determining MIS of " << argv[1] << " with " << xadj.extent(0)-1 << " nodes and " << adjncy.extent(0) << " edges using " << algorithm << "."<< std::endl;

            // Set up seed for RNG
            unsigned int seed;
            if(argc > 4){
                std::string str_seed= argv[4];
                seed = std::stoi(str_seed);
            } else{
                seed = (unsigned int)time(NULL);
            }

            for(int i = 0; i < 10; ++i){
                // Set up degrees
                Kokkos::View<int*> degree("degree", xadj.extent(0)-1);
                degree = initializeDegrees(xadj);
                
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
    }

    Kokkos::finalize();
    return 0;
}