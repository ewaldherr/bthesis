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

__global__ void removeVertices(int* graph,int* state, int n){
    if (state[threadIdx.x] == 1) {
        state[threadIdx.x] = 2;
        for (int j = 0; j < n; ++j) {
            if (graph[threadIdx.x+j*n] == 1) {
                state[j] = -1;
            }
        }
    }
}


// Luby's Algorithm with Kokkos
int* lubysAlgorithm(int* host_graph,int* host_state, int n) {
    int* graph;
    int* state;
    float* priorities;
    int* independentSet = new int[n];
    bool changes;
    int iters = 0;
    int* host_state = new int[n];
    curandState *d_state;

    cudaMalloc(&d_state, sizeof(curandState));
    cudaMalloc(&state,n*sizeof(int));
    cudaMalloc(&priorities,n*sizeof(float));
    cudaMalloc(&graph,n*n*sizeof(int));

    cudaMemcpy(state,host_state,n*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(graph,host_graph,n*n*sizeof(int),cudaMemcpyHostToDevice);

    do {
        initializePriorities<<<1,n>>>(priorities,d_state);
        checkMax<<<1,n>>>(graph,priorities,state,n);

        changes = false;
        cudaMemcpy(host_state,state,n*sizeof(int),cudaMemcpyDeviceToHost);
        for(int i = 0; i < n; ++i){
            if(host_state[i] == 1){
                changes = true;
                break;
            }
        }

        removeVertices<<<1,n>>>(graph,state,n);

        ++ iters;
    } while (changes);
    std::cout << iters << std::endl;

    cudaFree(state);
    cudaFree(priorities);
    cudaFree(graph);
    cudaFree(d_state);
    
    cudaMemcpy(independentSet,state, n*sizeof(int), cudaMemcpyDeviceToHost);
    return independentSet;
}

int main(int argc, char* argv[]) {
    {
        //Initialize graph
        int n = 6;
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
        // Run Luby's algorithm with Kokkos
        int* independentSet = new int[n];
        int* host_state = new int[n];
        for(int i = 0; i < n; ++i){
            host_state[i] = 0;
        }
        independentSet = lubysAlgorithm(adj,host_state,n);
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
