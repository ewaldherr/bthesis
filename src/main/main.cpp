#include "../read-write/output.cpp"
#include "../read-write/read_file.cpp"
#include "../algorithms/verify_result.cpp"
#include "../algorithms/luby_iter.cpp"

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

            // Run algorithm with Kokkos
            auto algo_start = std::chrono::high_resolution_clock::now();
            if(algorithm.compare("DEGREE") == 0){
                result_mis = degreeBasedAlgorithm(xadj,adjncy);
            } else if(algorithm.compare("LUBYITER") == 0){
                result_mis = LubyIterAlgorithm(xadj,adjncy,100);
            } else{
                Kokkos::View<int*> state("state", xadj.extent(0)-1);
                result_mis = lubysAlgorithm(xadj, adjncy, state);
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