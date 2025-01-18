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

int getSize(Kokkos::View<int*> mis){
    auto h_mis = Kokkos::create_mirror_view(mis);
    Kokkos::deep_copy(h_mis, mis);
    int size = 0;
    for(int i = 0; i < mis.extent(0); ++i){
        if(h_mis(i) == 1) size++;
    }
    std::cout << size << " vertices are inside of the MIS."  << std::endl;
    return size;
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
            std::cout << "Code running on " << Kokkos::DefaultExecutionSpace::name() << std::endl;
            // Set up seed for RNG
            unsigned int seed;
            if(argc > 2){
                std::string str_seed= argv[2];
                seed = std::stoi(str_seed);
            } else{
                seed = (unsigned int)time(NULL);
            }
            std::cout << "Using seed " << seed << std::endl;

            // Set up reduction
            std::string reductions[5] = {"NONE", "TRIVIAL", "ISOLATED", "DOMINATING", "ALL"};

            // Determining which algorithm to use
            std::string algorithms[3] = {"LUBY", "DEGREE", "DEGREEUD"};

            for(auto algo: algorithms){
                Kokkos::View<int*> result_mis("mis",xadj.extent(0)-1);
                std::cout << "Determining MIS of " << argv[1] << " with " << xadj.extent(0)-1 << " nodes and " << adjncy.extent(0)/2 << " edges using " << algo << "."<< std::endl;

                for(auto reduction: reductions){
                    int commulativeTime = 0;                  
                    double commulativeSize = 0;
                    for(int i = 0; i < 5; ++i){
                        // Set up degrees
                        Kokkos::View<int*> degree("degree", xadj.extent(0)-1);
                        degree = initializeDegrees(xadj);

                        // Run algorithm with Kokkos
                        Kokkos::View<int*> state("state", xadj.extent(0)-1);
                        Kokkos::deep_copy(state, -1);

                        // Execute reductions
                        std::cout << "Performing reduction " << reduction << std::endl;
                        int reduced = 0;
                        auto reduction_start = std::chrono::high_resolution_clock::now();
                        if(reduction.compare("NONE") != 0){
                            if(reduction.compare("TRIVIAL") == 0){
                                includeTrivial(degree,state,xadj,adjncy);
                            }
                            if(reduction.compare("ISOLATED") == 0){
                                includeIsolated(degree,state,xadj,adjncy);
                            }
                            if(reduction.compare("ALL") == 0){
                                allRed(degree,state,xadj,adjncy);
                            }
                            if(reduction.compare("DOMINATING") == 0){
                                removeDominating(degree,state,xadj,adjncy);
                            }
                            reduced = countAffected(state);
                            std::cout << "The reduction was conducted on " << reduced << " vertices" << std::endl;
                        }
                        auto reduction_stop = std::chrono::high_resolution_clock::now();
                        auto reduction_duration = std::chrono::duration_cast<std::chrono::microseconds>(reduction_stop - reduction_start);
                        std::cout << "Conducted reductions in " << reduction_duration.count() << " microseconds" << std::endl;

                        auto algo_start = std::chrono::high_resolution_clock::now();
                        if(algo.compare("DEGREE") == 0 || algo.compare("DEGREEUD") == 0){
                            result_mis = degreeBasedAlgorithm(xadj, adjncy, degree, state, seed + 100 * i, algo, 1);
                        } else if(algo.compare("LUBYITER") == 0 || algo.compare("DEGREEITER") == 0){
                            result_mis = iterAlgorithm(xadj, adjncy, degree, algo, seed + 100 * i);
                        } else{
                            result_mis = lubysAlgorithm(xadj, adjncy, state, seed + 100 * i);
                        }
                        auto algo_stop = std::chrono::high_resolution_clock::now();
                        auto algo_duration = std::chrono::duration_cast<std::chrono::microseconds>(algo_stop - algo_start);
                        commulativeTime += algo_duration.count() + reduction_duration.count();
                        std::cout << "Determined MIS in " << algo_duration.count() + reduction_duration.count() << " microseconds" << std::endl;
                        commulativeSize += getSize(result_mis);
                        std::cout << "Verifying solution..." << std::endl;
                        bool valid = verifyResult(result_mis, xadj, adjncy);
                        if(valid){
                            std::cout << "Solution is valid" << std::endl;
                        } else{
                            std::cout << "Solution is NOT valid" << std::endl;
                        }
                    }

                    std::cout << "Average solution size is " << commulativeSize / 5 << std::endl;
                    std::cout << "Average execution time is " << commulativeTime / 5 << std::endl;

                }

            }
        }
    }
    Kokkos::finalize();
    return 0;
}