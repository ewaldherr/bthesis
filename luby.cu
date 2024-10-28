#include <iostream>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>

__global__ void initializePriorities(float* priorities,curandState * d_state) {
    curand_init(1234, threadIdx.x, 0, &d_state[threadIdx.x]);
    priorities[threadIdx.x] = curand_uniform(d_state + threadIdx.x);
}

__global__ void checkMax(int* removed, int** graph,float* priorities,int* inMIS, int n){
    if (removed[threadIdx.x] == 1) return;
    bool isMaxPriority = true;
        for (int j = 0; j < n; ++j) {
            if (graph[threadIdx.x][j] == 1 && removed[j] == 0 && priorities[j] >= priorities[threadIdx.x]) {
                isMaxPriority = false;
                break;
            }
        }

        if (isMaxPriority) {
                inMIS[threadIdx.x] = 1;
        }
}

__global__ void removeVertices(int* removed, int** graph,int* inMIS,bool& changes, int n){
    if (inMIS[threadIdx.x] == 1) {
        removed[threadIdx.x] = 1;
        for (int j = 0; j < n; ++j) {
            if (graph[threadIdx.x][j] == 1) {
                removed[j] = 1;
            }
        }
        changes = true; // If any vertex is added, flag a change
    }
}


// Luby's Algorithm with Kokkos
int* lubysAlgorithm(int* removed, int** graph,float* priorities,int* inMIS, int n) {
    int* independentSet = new int[n];
    bool changes;
    bool d_changes;
    cudaMalloc(&d_changes, sizeof(bool));
    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));
    do {
        // Step 1: Assign random priorities to remaining vertices
        initializePriorities<<<1,n>>>(priorities,d_state);
        checkMax<<<1,n>>>(removed,graph,priorities,inMIS,n);
        // Step 3: Add selected vertices to MIS and remove them and their neighbors
        changes = false;
        cudaMemcpy(d_changes,changes,sizeof(bool),cudaMemcpyHostToDevice);
        removeVertices<<<1,n>>>(removed,graph,inMIS,changes,n);
        cudaMemcpy(changes,d_changes,sizeof(bool),cudaMemcpyDeviceToHost);
    } while (changes);
    cudaMemcpy(independentSet,inMIS, n*sizeof(int), cudaMemcpyDeviceToHost);
    return independentSet;
}

int main(int argc, char* argv[]) {
    {
        //Initialize graph
        int n = 6;
        int** adj = new int*[n];
        for(int i = 0;i < n; ++i){
            adj[i] = new int[n];
            for(int j = 0;j < n; ++j){
                adj[i][j] = 0;
            }
        }
        adj[0][1] = 1;
        adj[0][2] = 1;
        adj[1][3] = 1;
        adj[2][3] = 1;
        adj[3][4] = 1;
        adj[3][5] = 1;
        // Run Luby's algorithm with Kokkos
        int** d_adj;
        int* inMIS = new int[n];
        int* removed = new int[n];
        float* priorities = new float[n];
        int* independentSet = new int[n];
        cudaMalloc(&inMIS,n*sizeof(int));
        cudaMalloc(&removed,n*sizeof(int));
        cudaMalloc(&priorities,n*sizeof(float));
        cudaMalloc(&d_adj,n*n*sizeof(int));
        cudaMemcpy(d_adj,adj,n*n*sizeof(int),cudaMemcpyHostToDevice);
        independentSet = lubysAlgorithm(removed,adj,priorities,inMIS,n);
        cudaFree(inMIS);
        cudaFree(removed);
        cudaFree(priorities);
        cudaFree(d_adj);
        // Print the result
        std::cout << "Maximum Independent Set (MIS) nodes: " << std::endl;
        for(int i = 0; i < n; ++i){
            if (independentSet[i] == 1) {
                std::cout << i << " ";
            }
        }
        std::cout << std::endl;

    }
    return 0;
}
