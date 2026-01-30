#include <thrust/sort.h>
#include "gpu_dbscan.h"
#include "common.h"
#include "helper.h"
#include "cuda_helper.h"

#define BLOCK_SIZE 128

__constant__ float cXMin;
__constant__ float cYMin;
__constant__ int cGridWidth;
__constant__ int cGridHeight;
__constant__ int cCellCount;
__constant__ int cN;
__constant__ float cEps;
__constant__ float cInvEps;
__constant__ float cEps2;
__constant__ int cMinPts;

__global__ void computeBinning(
    int *cellId,
    int *cellPoints,
    const float *x,
    const float *y
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= cN) return;

    int cx, cy;
    pointCellCoordinates(
        &cx, &cy,
        x[tid], y[tid],
        cXMin, cYMin, cInvEps
    );

    cellId[tid] = linearCellId(cx, cy, cGridWidth);
    cellPoints[tid] = tid;
}

__global__ void buildBinExtreme(
    int *cellStart,
    int *cellEnd,
    const int *cellId
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warpSize;

    if (tid >= cN) return;

    const int cid = cellId[tid];

    int prevCid = __shfl_up_sync(0xffffffff, cid, 1, warpSize);
    // The first thread in the warp need to get the previous cid directly from the global memory
    if (lane == 0 && tid > 0) {
        prevCid = cellId[tid - 1];
    }

    int nextCid = __shfl_down_sync(0xffffffff, cid, 1, warpSize);
    // The last thread in the warp need to get the next cid directly from the global memory
    if (lane == warpSize - 1 && tid + 1 < cN) {
        nextCid = cellId[tid + 1];
    }

    if (tid == 0 || cid != prevCid) {
        cellStart[cid] = tid;
    }
    if (tid == cN - 1 || cid != nextCid) {
        cellEnd[cid] = tid + 1;
    }
}

__global__ void computeIsCoreArr(
    bool *isCoreArr,
    const float *x,
    const float *y,
    const int *cellStart,
    const int *cellEnd,
    const int *cellPoints
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= cN) return;

    const float xi = x[tid];
    const float yi = y[tid];

    int cx, cy;
    pointCellCoordinates(
        &cx, &cy,
        x[tid], y[tid],
        cXMin, cYMin, cInvEps
    );

    int count = 0;

    #pragma unroll
    for (int idx = 0; idx < 9; ++idx) {
        const int dx = idx / 3 - 1;
        const int dy = idx % 3 - 1;

        const int nx = cx + dx;
        const int ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= cGridWidth || ny >= cGridHeight) continue;

        const int nid = linearCellId(nx, ny, cGridWidth);
        const int start = cellStart[nid];
        const int end = cellEnd[nid];

        for (int k = start; k < end; k++) {
            const int j = cellPoints[k];

            if (tid != j && isEpsNeighbor(xi, yi, x[j], y[j], cEps2) &&
                ++count >= cMinPts) {
                isCoreArr[tid] = true;
                return;
            }
        }
    }

    isCoreArr[tid] = isCore(count, cMinPts);
}

__global__ void bfsExpand(
    int *cluster,
    int *frontier,
    int *nextFrontier,
    int *nextFrontierSize,
    const float *x,
    const float *y,
    const int *cellStart,
    const int *cellEnd,
    const int *cellPoints,
    const bool *isCoreArr,
    const int frontierSize,
    const int clusterId,
    const int level,
    const int root
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Assign cluster to core point
    if (tid == 0 && level == 0) {
        cluster[root] = clusterId;
        frontier[0] = root;
    }

    __syncthreads();

    if (tid >= frontierSize) return;

    const int i = frontier[tid];
    const float xi = x[i];
    const float yi = y[i];

    int cx, cy;
    pointCellCoordinates(
        &cx, &cy,
        xi, yi,
        cXMin, cYMin, cInvEps
    );

    #pragma unroll
    for (int idx = 0; idx < 9; ++idx) {
        const int dx = idx / 3 - 1;
        const int dy = idx % 3 - 1;

        const int nx = cx + dx;
        const int ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= cGridWidth || ny >= cGridHeight) continue;

        const int nid = linearCellId(nx, ny, cGridWidth);
        const int start = cellStart[nid];
        const int end = cellEnd[nid];

        for (int k = start; k < end; k++) {
            const int j = cellPoints[k];

            // Add neighbor to frontier iff it is not yet assigned to a cluster and is a core point
            if (i != j && isEpsNeighbor(xi, yi, x[j], y[j], cEps2) &&
                atomicCAS(&cluster[j], NO_CLUSTER_LABEL, clusterId) == NO_CLUSTER_LABEL &&
                isCoreArr[j]) {
                const int pos = atomicAdd(nextFrontierSize, 1);
                nextFrontier[pos] = j;
            }
        }
    }
}

void dbscan_gpu(
    int *cluster,
    int *clusterCount,
    const float *x,
    const float *y,
    const int n,
    const float eps,
    const int minPts
) {
    // int device;
    // cudaDeviceProp prop;
    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&prop, device);

    float xMin = x[0], xMax = x[0];
    float yMin = y[0], yMax = y[0];
    for (int i = 1; i < n; i++) {
        xMin = fmin(xMin, x[i]);
        xMax = fmax(xMax, x[i]);
        yMin = fmin(yMin, y[i]);
        yMax = fmax(yMax, y[i]);
    }

    const int gridWidth = ceil((xMax - xMin) / eps + 1);
    const int gridHeight = ceil((yMax - yMin) / eps + 1);
    const int cellCount = gridWidth * gridHeight;
    const float invEps = 1.0 / eps;
    const float eps2 = eps * eps;

    bool *hIsCoreArr = (bool *) malloc_s(n * sizeof(bool));
    if (!hIsCoreArr) return;

    int *dCluster;
    float *dX, *dY;
    int *dCellId;
    int *dCellStart;
    int *dCellEnd;
    int *dCellPoints;
    bool *dIsCoreArr;
    int *dFrontier;
    int *dNextFrontier;
    int *dNextFrontierSize;

    CUDA_CHECK(cudaMalloc(&dCluster, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dX, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCellId, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCellStart, cellCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCellEnd, cellCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCellPoints, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dIsCoreArr, n * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&dFrontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNextFrontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNextFrontierSize, sizeof(int)));

    CUDA_CHECK(cudaMemcpyToSymbol(cXMin, &xMin, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cYMin, &yMin, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cGridWidth, &gridWidth, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(cGridHeight, &gridHeight, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(cCellCount, &cellCount, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(cN, &n, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(cEps, &eps, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cInvEps, &invEps, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cEps2, &eps2, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cMinPts, &minPts, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dX, x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, y, n * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dCluster, NO_CLUSTER_LABEL, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(dCellStart, 0, cellCount * sizeof(int)));
    CUDA_CHECK(cudaMemset(dCellEnd, 0, cellCount * sizeof(int)));

    launchKernel(
        computeBinning, nullptr, n,
        dCellId, dCellPoints, dX, dY
    );

    thrust::sort_by_key(
        thrust::device,
        dCellId,
        dCellId + n,
        dCellPoints
    );

    launchKernel(
        buildBinExtreme, nullptr, n,
        dCellStart, dCellEnd, dCellId
    );

    launchKernel(
        computeIsCoreArr, nullptr, n,
        dIsCoreArr, dX, dY, dCellStart, dCellEnd, dCellPoints
    );

    CUDA_CHECK(cudaMemcpy(hIsCoreArr, dIsCoreArr, n * sizeof(bool), cudaMemcpyDeviceToHost));

    int clusterId = NO_CLUSTER_LABEL;
    dim3 block(BLOCK_SIZE);

    for (int p = 0; p < n; p++) {
        if (!hIsCoreArr[p]) continue;

        int hCluster;
        CUDA_CHECK(cudaMemcpy(&hCluster, dCluster + p, sizeof(int), cudaMemcpyDeviceToHost));
        if (hCluster != NO_CLUSTER_LABEL) continue;

        clusterId++;

        int level = 0;
        int frontier_size = 1;
        while (frontier_size > 0) {
            CUDA_CHECK(cudaMemset(dNextFrontierSize, 0, sizeof(int)));

            dim3 bfs_grid((frontier_size + block.x - 1) / block.x);

            bfsExpand<<<bfs_grid, block>>>(
                dCluster, dFrontier, dNextFrontier,
                dNextFrontierSize, dX, dY, dCellStart,
                dCellEnd, dCellPoints, dIsCoreArr,
                frontier_size, clusterId, level, p
            );

            level++;
            CUDA_CHECK(cudaMemcpy(&frontier_size, dNextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost));
            std::swap(dFrontier, dNextFrontier);
        }
    }

    *clusterCount = clusterId;

    CUDA_CHECK(cudaMemcpy(cluster, dCluster, n * sizeof(int), cudaMemcpyDeviceToHost));

    free(hIsCoreArr);
    CUDA_CHECK(cudaFree(dCluster));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dCellId));
    CUDA_CHECK(cudaFree(dCellStart));
    CUDA_CHECK(cudaFree(dCellEnd));
    CUDA_CHECK(cudaFree(dCellPoints));
    CUDA_CHECK(cudaFree(dIsCoreArr));
    CUDA_CHECK(cudaFree(dFrontier));
    CUDA_CHECK(cudaFree(dNextFrontier));
    CUDA_CHECK(cudaFree(dNextFrontierSize));
}
