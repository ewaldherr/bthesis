#include <iostream>
#include <vector>

__global__ void initializePriorities(std::vector<int>& priorities) {
    unsigned seed = 1234 + threadIdx.x; // Seed for random number generator
    priorities(threadIdx.x) = rand_r(&seed);
}

__global__ void checkMax(std::vector<int>& removed, std::vector<std::vector<int>>& graph,std::vector<int>& priorities,std::vector<int>& inMIS){
    if (removed(threadIdx.x) == 1) return;
    bool isMaxPriority = true;
        for (int j = 0; j < graph.size(); ++j) {
            if (graph[threadIdx.x][j] == 1 && removed(j) == 0 && priorities(j) >= priorities(threadIdx)) {
                isMaxPriority = false;
                break;
            }
        }

        if (isMaxPriority) {
                inMIS(threadIdx.x) = 1;
        }
}

__global__ void removeVertices(std::vector<int>& removed, std::vector<std::vector<int>>& graph,std::vector<int>& inMIS,bool& changes){
    if (inMIS(threadIdx.x) == 1) {
        removed(threadIdx.x) = 1;
        for (int j = 0; j < graph.size(); ++j) {
            if (graph(threadIdx.x, j) == 1) {
                removed(j) = 1;
            }
        }
        changes = true; // If any vertex is added, flag a change
    }
}


// Luby's Algorithm with Kokkos
std::vector<int> lubysAlgorithm(std::vector<int>& removed, std::vector<std::vector<int>>& graph,std::vector<int>& priorities,std::vector<int>& inMIS) {
    std::vector<int> independentSet(graph.size());
    bool changes;
    do {
        // Step 1: Assign random priorities to remaining vertices
        initializePriorities<<<1,graph.size()>>>(priorities);
        checkMax<<<1,graph.size()>>>(removed,graph,priorities,inMIS);
        // Step 3: Add selected vertices to MIS and remove them and their neighbors
        changes = false;
        removeVertices<<<1,graph.size()>>>(removed,graph,inMIS,&changes);
    } while (changes);
    cudaMemcpy(independentSet,inMIS, n*sizeof(int), cudaMemcpyDeviceToHost);
    return independentSet;
}

int main(int argc, char* argv[]) {
    {
        //Initialize graph
        int n = 6;
        std::vector<std::vector<int>> adj(n);
        for(int i = 0;i < n; ++i){
            for(int j = 0;j < n; ++j){
                adj[i].push_back(0);
            }
        }
        adj[0][1] = 1;
        adj[0][2] = 1;
        adj[1][3] = 1;
        adj[2][3] = 1;
        adj[3][4] = 1;
        adj[3][5] = 1;
        // Run Luby's algorithm with Kokkos
        std::vector<std::vector<int>> d_adj;
        std::vector<int> inMIS(graph.size());
        std::vector<int> removed(graph.size());
        std::vector<int> priorities(graph.size());
        std::vector<int> independentSet;
        cudaMalloc(&inMIS,n*sizeof(int));
        cudaMalloc(&removed,n*sizeof(int));
        cudaMalloc(&priorities,n*sizeof(int));
        cudaMalloc(&d_adj,n*n*sizeof(int));
        cudaMemcpy(d_adj,adj,n*n*sizeof(int),cudaMemcpyHostToDevice);
        stid::vector<int> independentSet = lubysAlgorithm(removed,adj,priorities,inMIS);
        cudaFree(inMIS);
        cudaFree(removed);
        cudaFree(priorities);
        cudaFree(d_adj);
        // Print the result
        std::cout << "Maximum Independent Set (MIS) nodes:" << std::endl;
        for(int i = 0; i < n; ++i){
            if (independentSet(i) == 1) {
                printf("%d ", i);
            }
        }
        std::cout << std::endl;

    }
    return 0;
}
