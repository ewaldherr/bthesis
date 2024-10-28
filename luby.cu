#include <iostream>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>

__global__ void initializePriorities(float* priorities,curandState * d_state) {
    curand_init(1234, threadIdx.x, 0, &d_state[threadIdx.x]);
    priorities[threadIdx.x] = curand_uniform(d_state + threadIdx.x);
}

__global__ void checkMax(int* graph,float* priorities,int* state, int n){
    if (state[threadIdx.x] != 0) return;
    bool isMaxPriority = true;
        for (int j = 0; j < n; ++j) {
            if (graph[threadIdx.x+j*n] == 1 && state[j] == 0 && priorities[j] >= priorities[threadIdx.x]) {
                isMaxPriority = false;
                break;
            }
        }

        if (isMaxPriority) {
                state[threadIdx.x] = 1;
        }
}

__global__ void removeVertices(int* graph,int* state,bool* changes, int n){
    if (state[threadIdx.x] == 1) {
        state[threadIdx.x] = 2;
        for (int j = 0; j < n; ++j) {
            if (graph[threadIdx.x+j*n] == 1) {
                state[j] = -1;
            }
        }
        changes[0] = true; // If any vertex is added, flag a change
    }
}


// Luby's Algorithm with Kokkos
int* lubysAlgorithm(int* graph,float* priorities,int* state, int n) {
    int* host_adj = new int [n*n];
    cudaMemcpy(host_adj,graph,n*n*sizeof(int),cudaMemcpyDeviceToHost);
    for (int i=0;i<n;++i){
        std::cout << std::endl;
        for(int j=0;j<n;++j){
            std::cout << host_adj [i+j*n] << " ";
        }
    }
    std::cout << std::endl;
    float* host_prios = new float[n];
    int* host_state = new int[n];
    int* independentSet = new int[n];
    bool* changes = new bool[1];
    bool* d_changes = new bool[1];
    cudaMalloc(&d_changes, sizeof(bool));
    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));
    int iters = 0;
    do {
        // Step 1: Assign random priorities to remaining vertices
        initializePriorities<<<1,n>>>(priorities,d_state);
        cudaMemcpy(host_prios,priorities,n*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(host_state,state,n*sizeof(int),cudaMemcpyDeviceToHost);
        for (int i = 0; i< n; ++i){
            	std::cout << host_prios[i] << " " << host_state[i] << "  ";
        }
        std::cout << std::endl;
        checkMax<<<1,n>>>(graph,priorities,state,n);
        cudaMemcpy(host_state,state,n*sizeof(int),cudaMemcpyDeviceToHost);
        for (int i = 0; i< n; ++i){
            	std::cout << host_prios[i] << " " << host_state[i] << "  ";
        }
        std::cout << std::endl;
        // Step 3: Add selected vertices to MIS and remove them and their neighbors
        changes[0] = false;
        cudaMemcpy(d_changes,changes,sizeof(bool),cudaMemcpyHostToDevice);
        removeVertices<<<1,n>>>(graph,state,changes,n);
        cudaMemcpy(host_state,state,n*sizeof(int),cudaMemcpyDeviceToHost);
        for (int i = 0; i< n; ++i){
            	std::cout << host_prios[i] << " " << host_state[i] << "  ";
        }
        std::cout << std::endl;
        cudaMemcpy(changes,d_changes,sizeof(bool),cudaMemcpyDeviceToHost);
        ++ iters;
    } while (changes[0]);
    std::cout << iters << std::endl;
    cudaMemcpy(independentSet,state, n*sizeof(int), cudaMemcpyDeviceToHost);
    return independentSet;
}

int main(int argc, char* argv[]) {
    {
        //Initialize graph
        int n = 6;
        #define N n;
        int* adj = new int[n*n];
        for(int i = 0;i < n; ++i){
            for(int j = 0;j < n; ++j){
                adj[i+j*n] = 0;
            }
        }
        adj[0 + 1 * 6] = 1;
        adj[0 + 2 * 6] = 1;
        adj[1 + 3 * 6] = 1;
        adj[2 + 3 * 6] = 1;
        adj[3 + 4 * 6] = 1;
        adj[3 + 5 * 6] = 1;
        //backward edges
        adj[1 + 0 * 6] = 1;
        adj[2 + 0 * 6] = 1;
        adj[3 + 1 * 6] = 1;
        adj[3 + 2 * 6] = 1;
        adj[4 + 3 * 6] = 1;
        adj[5 + 3 * 6] = 1;
        for (int i=0;i<n;++i){
        std::cout << std::endl;
        for(int j=0;j<n;++j){
            std::cout << adj [i+j*n] << " ";
        }
    }
        // Run Luby's algorithm with Kokkos
        int (*d_adj)[n];
        int* host_state = new int[n];
        for(int i = 0; i < n; ++i){
            host_state[i] = 0;
        }
        int* state = new int[n];
        float* priorities = new float[n];
        int* independentSet = new int[n];
        cudaMalloc(&state,n*sizeof(int));
        cudaMalloc(&priorities,n*sizeof(float));
        cudaMalloc((void**)&d_adj,n*n*sizeof(int));
        cudaMemcpy(state,host_state,n*sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_adj,adj,n*n*sizeof(int),cudaMemcpyHostToDevice);
        independentSet = lubysAlgorithm(adj,priorities,state,n);
        cudaFree(state);
        cudaFree(priorities);
        cudaFree(d_adj);
        // Print the result
        std::cout << "Maximum Independent Set (MIS) nodes: " << std::endl;
        for(int i = 0; i < n; ++i){
            if (independentSet[i] == 2) {
                std::cout << i << " ";
            }
        }
        std::cout << std::endl;

    }
    return 0;
}
