#include <stdio.h>
#include <cuda_runtime.h>
#include <set>

#define MAX_VERTICES 1024
#define MAX_EDGES 5000

// Kernel to calculate initial triangle support
__global__ void calculateInitialSupport(int *adj, int *edges, int *sup, int numEdges, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        int u = edges[idx * 2];
        int v = edges[idx * 2 + 1];
        int count = 0;
        for (int i = 0; i < numVertices; i++) {
            if (adj[u * numVertices + i] && adj[v * numVertices + i]) {
                count++;
            }
        }
        sup[idx] = count;
    }
}

// Kernel to update support for affected edges
__global__ void updateSupport(int *adj, int *edges, int *sup, int *setDel, int setDelSize, int numEdges, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < setDelSize) {
        int eIdx = setDel[idx];
        int u = edges[eIdx * 2];
        int v = edges[eIdx * 2 + 1];
        for (int w = 0; w < numVertices; w++) {
            if (adj[u * numVertices + w] && adj[v * numVertices + w]) {
                for (int e = 0; e < numEdges; e++) {
                    int x = edges[e * 2];
                    int y = edges[e * 2 + 1];
                    if ((x == u && y == w) || (x == w && y == u)) {
                        if (sup[e] > 0) sup[e]--;
                    }
                    if ((x == v && y == w) || (x == w && y == v)) {
                        if (sup[e] > 0) sup[e]--;
                    }
                }
            }
        }
    }
}

void optimizedKTruss(int *adj, int *edges, int numVertices, int numEdges, int k, int *edgeDel, int *numTrusses, int *remainingEdges, int *totalTriangles) {
    int *d_adj, *d_edges, *d_sup, *d_setDel;
    cudaMalloc(&d_adj, numVertices * numVertices * sizeof(int));
    cudaMalloc(&d_edges, numEdges * 2 * sizeof(int));
    cudaMalloc(&d_sup, numEdges * sizeof(int));
    cudaMalloc(&d_setDel, numEdges * sizeof(int));

    cudaMemcpy(d_adj, adj, numVertices * numVertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, edges, numEdges * 2 * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numEdges + blockSize - 1) / blockSize;

    // Initial support calculation
    calculateInitialSupport<<<numBlocks, blockSize>>>(d_adj, d_edges, d_sup, numEdges, numVertices);
    cudaDeviceSynchronize();

    int *sup = (int *)malloc(numEdges * sizeof(int));
    cudaMemcpy(sup, d_sup, numEdges * sizeof(int), cudaMemcpyDeviceToHost);

    std::set<int> setDel;
    for (int e = 0; e < numEdges; e++) {
        if (edgeDel[e] == -1 && sup[e] < k - 2) {
            edgeDel[e] = k - 1;
            setDel.insert(e);
        }
    }

    while (!setDel.empty()) {
        int setDelSize = setDel.size();
        int *setDelArray = (int *)malloc(setDelSize * sizeof(int));
        int i = 0;
        for (int e : setDel) setDelArray[i++] = e;

        cudaMemcpy(d_setDel, setDelArray, setDelSize * sizeof(int), cudaMemcpyHostToDevice);
        numBlocks = (setDelSize + blockSize - 1) / blockSize;
        updateSupport<<<numBlocks, blockSize>>>(d_adj, d_edges, d_sup, d_setDel, setDelSize, numEdges, numVertices);
        cudaDeviceSynchronize();

        cudaMemcpy(sup, d_sup, numEdges * sizeof(int), cudaMemcpyDeviceToHost);
        setDel.clear();
        for (int e = 0; e < numEdges; e++) {
            if (edgeDel[e] == -1 && sup[e] < k - 2) {
                edgeDel[e] = k - 1;
                setDel.insert(e);
            }
        }
        free(setDelArray);
    }

    // Post-processing to count trusses, remaining edges, and triangles
    *remainingEdges = 0;
    int *ktrussAdj = (int *)malloc(numVertices * numVertices * sizeof(int));
    memset(ktrussAdj, 0, numVertices * numVertices * sizeof(int));
    for (int e = 0; e < numEdges; e++) {
        if (edgeDel[e] == -1) {
            (*remainingEdges)++;
            int u = edges[e * 2];
            int v = edges[e * 2 + 1];
            ktrussAdj[u * numVertices + v] = 1;
            ktrussAdj[v * numVertices + u] = 1;
        }
    }
    *totalTriangles = 0;
    for (int u = 0; u < numVertices; u++) {
        for (int v = u + 1; v < numVertices; v++) {
            if (ktrussAdj[u * numVertices + v]) {
                for (int w = v + 1; w < numVertices; w++) {
                    if (ktrussAdj[v * numVertices + w] && ktrussAdj[u * numVertices + w]) {
                        (*totalTriangles)++;
                    }
                }
            }
        }
    }
    *numTrusses = (*remainingEdges > 0) ? 1 : 0;

    free(sup);
    free(ktrussAdj);
    cudaFree(d_adj);
    cudaFree(d_edges);
    cudaFree(d_sup);
    cudaFree(d_setDel);
}

int main() {
    srand(time(NULL));
    int numVertices = 100;
    int numEdges = 0;
    int adj[MAX_VERTICES * MAX_VERTICES];
    int edges[MAX_EDGES * 2];
    int edgeDel[MAX_EDGES];

    // Initialize adjacency matrix with 70% density
    for (int i = 0; i < numVertices * numVertices; i++) {
        int row = i / numVertices;
        int col = i % numVertices;
        adj[i] = (rand() % 100 < 70) ? 1 : 0; // 70% edge probability
        if (row == col) adj[i] = 0; // No self-loops
        adj[col * numVertices + row] = adj[i]; // Symmetry
    }

    // Generate edge list
    for (int i = 0; i < numVertices; i++) {
        for (int j = i + 1; j < numVertices; j++) {
            if (adj[i * numVertices + j]) {
                if (numEdges < MAX_EDGES) {
                    edges[numEdges * 2] = i;
                    edges[numEdges * 2 + 1] = j;
                    edgeDel[numEdges] = -1;
                    numEdges++;
                }
            }
        }
    }

    int k = 4;
    int numTrusses = 0, remainingEdges = 0, totalTriangles = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    optimizedKTruss(adj, edges, numVertices, numEdges, k, edgeDel, &numTrusses, &remainingEdges, &totalTriangles);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Optimized k-Truss Time: %f ms\n", milliseconds);
    printf("Number of %d-Trusses: %d\n", k, numTrusses);
    printf("Remaining Edges in %d-Truss: %d\n", k, remainingEdges);
    printf("Total Triangles in %d-Truss: %d\n", k, totalTriangles);
    printf("\nEdge Deletion Status (first 10 edges as sample):\n");
    for (int i = 0; i < (numEdges > 10 ? 10 : numEdges); i++) {
        printf("Edge (%d, %d): %s\n", edges[i * 2], edges[i * 2 + 1], edgeDel[i] == -1 ? "Kept" : "Deleted");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
