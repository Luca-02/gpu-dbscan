#include <thrust/sort.h>
#include "gpu_dbscan.h"
#include "common.h"
#include "helper.h"
#include "cuda_helper.h"

#define BLOCK_SIZE 512

__constant__ double cXMin;
__constant__ double cYMin;
__constant__ int cGridWidth;
__constant__ int cGridHeight;
__constant__ int cCellCount;
__constant__ int cN;
__constant__ double cEps;
__constant__ double cInvEps;
__constant__ double cEps2;
__constant__ int cMinPts;

size_t smemForBinning(const int blockSize) {
    return blockSize * 2 * sizeof(double);
}

size_t smemForBuildBinExtreme(const int blockSize) {
    return blockSize / 32 * 2 * sizeof(int);
}

__global__ void computeBinning(
    int *cellId,
    int *cellPoints,
    const double *x,
    const double *y
) {
    extern __shared__ double pointsSmem[];
    double *sX = pointsSmem;
    double *sY = pointsSmem + blockDim.x;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < cN) {
        sX[threadIdx.x] = x[tid];
        sY[threadIdx.x] = y[tid];
    }

    __syncthreads();

    if (tid < cN) {
        int cx, cy;
        pointCellCoordinates(
            &cx, &cy,
            sX[threadIdx.x], sY[threadIdx.x],
            cXMin, cYMin, cInvEps
        );

        cellId[tid] = linearCellId(cx, cy, cGridWidth);
        cellPoints[tid] = tid;
    }
}

__global__ void buildBinExtreme(
    int *cellStart,
    int *cellEnd,
    const int *cellIdSorted
) {
    extern __shared__ int cellIdSortedSmem[];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warpSize;
    const int wid = threadIdx.x / warpSize;

    if (lane == 0 && tid > 0) {
        cellIdSortedSmem[wid * 2] = cellIdSorted[tid - 1];
    }

    if (lane == warpSize - 1 && tid + 1 < cN) {
        cellIdSortedSmem[wid * 2 + 1] = cellIdSorted[tid + 1];
    }

    __syncthreads();

    if (tid < cN) {
        const int cid = cellIdSorted[tid];

        int prevCid = __shfl_up_sync(0xffffffff, cid, 1, warpSize);
        // The first thread in the warp need to get the previous cid directly from the global memory
        if (lane == 0 && tid > 0) {
            prevCid = cellIdSortedSmem[wid * 2];
        }

        int nextCid = __shfl_down_sync(0xffffffff, cid, 1, warpSize);
        // The last thread in the warp need to get the next cid directly from the global memory
        if (lane == warpSize - 1 && tid + 1 < cN) {
            nextCid = cellIdSortedSmem[wid * 2 + 1];
        }

        if (tid == 0 || cid != prevCid) {
            cellStart[cid] = tid;
        }
        if (tid == cN - 1 || cid != nextCid) {
            cellEnd[cid] = tid + 1;
        }
    }
}

__global__ void computeNeighborCount(
    int *neighborCount,
    const double *x,
    const double *y,
    const int *cellId,
    const int *cellStart,
    const int *cellEnd,
    const int *cellPoints
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cN) return;

    const double xi = x[tid];
    const double yi = y[tid];

    const int cid = cellId[tid];
    const int cx = cid % cGridWidth;
    const int cy = cid / cGridWidth;

    int count = 0;

#pragma unroll
    for (int dx = -1; dx <= 1; dx++) {
#pragma unroll
        for (int dy = -1; dy <= 1; dy++) {
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= cGridWidth || ny >= cGridHeight) continue;

            const int nid = linearCellId(nx, ny, cGridWidth);
            const int start = cellStart[nid];
            const int end = cellEnd[nid];

            for (int k = start; k < end; k++) {
                const int j = cellPoints[k];

                if (tid != j && isEpsNeighbor(xi, yi, x[j], y[j], cEps2)) {
                    count++;
                }
            }
        }
    }

    neighborCount[tid] = count;
}

__global__ void bfsExpand(
    int *cluster,
    int *nextFrontier,
    int *nextFrontierSize,
    const double *x,
    const double *y,
    const int *cellStart,
    const int *cellEnd,
    const int *cellPoints,
    const int *neighborCount,
    const int *frontier,
    const int frontierSize,
    const int clusterId
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontierSize) return;

    const int i = frontier[tid];
    const double xi = x[i];
    const double yi = y[i];

    int cx, cy;
    pointCellCoordinates(
        &cx, &cy,
        xi, yi,
        cXMin, cYMin, cInvEps
    );

#pragma unroll
    for (int dx = -1; dx <= 1; dx++) {
#pragma unroll
        for (int dy = -1; dy <= 1; dy++) {
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= cGridWidth || ny >= cGridHeight) continue;

            const int nid = linearCellId(nx, ny, cGridWidth);
            const int start = cellStart[nid];
            const int end = cellEnd[nid];

            for (int k = start; k < end; k++) {
                const int j = cellPoints[k];

                if (i != j && isEpsNeighbor(xi, yi, x[j], y[j], cEps2)) {
                    const int old = atomicCAS(&cluster[j], NO_CLUSTER_LABEL, clusterId);

                    // Add neighbor to frontier iff it is not yet assigned to a cluster and is a core point
                    if (old == NO_CLUSTER_LABEL && isCore(neighborCount[j], cMinPts)) {
                        const int pos = atomicAdd(nextFrontierSize, 1);
                        nextFrontier[pos] = j;
                    }
                }
            }
        }
    }
}

void dbscan_gpu(
    int *cluster,
    int *clusterCount,
    const double *x,
    const double *y,
    const int n,
    const double eps,
    const int minPts
) {
    // int device;
    // cudaDeviceProp prop;
    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&prop, device);

    double xMin = x[0], xMax = x[0];
    double yMin = y[0], yMax = y[0];
    for (int i = 1; i < n; i++) {
        xMin = fmin(xMin, x[i]);
        xMax = fmax(xMax, x[i]);
        yMin = fmin(yMin, y[i]);
        yMax = fmax(yMax, y[i]);
    }

    const int gridWidth = ceil((xMax - xMin) / eps + 1);
    const int gridHeight = ceil((yMax - yMin) / eps + 1);
    const int cellCount = gridWidth * gridHeight;
    const double invEps = 1.0 / eps;
    const double eps2 = eps * eps;

    int *hNeighborCount = (int *) malloc_s(n * sizeof(int));
    if (!hNeighborCount) return;

    int *dCluster;
    double *dX, *dY;
    int *dCellId;
    int *dCellIdSorted;
    int *dCellStart;
    int *dCellEnd;
    int *dCellPoints;
    int *dNeighborCount;
    int *dFrontier;
    int *dNextFrontier;
    int *dNextFrontierSize;

    CUDA_CHECK(cudaMalloc(&dCluster, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dX, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dY, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCellId, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCellIdSorted, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCellStart, cellCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCellEnd, cellCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCellPoints, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNeighborCount, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dFrontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNextFrontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dNextFrontierSize, sizeof(int)));

    CUDA_CHECK(cudaMemcpyToSymbol(cXMin, &xMin, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(cYMin, &yMin, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(cGridWidth, &gridWidth, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(cGridHeight, &gridHeight, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(cCellCount, &cellCount, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(cN, &n, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(cEps, &eps, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(cInvEps, &invEps, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(cEps2, &eps2, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(cMinPts, &minPts, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dX, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, y, n * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dCluster, NO_CLUSTER_LABEL, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(dCellStart, 0, cellCount * sizeof(int)));
    CUDA_CHECK(cudaMemset(dCellEnd, 0, cellCount * sizeof(int)));

    launchKernel(
        computeBinning, smemForBinning, n,
        dCellId, dCellPoints, dX, dY
    );

    CUDA_CHECK(cudaMemcpy(dCellIdSorted, dCellId, n * sizeof(int), cudaMemcpyDeviceToDevice));

    thrust::sort_by_key(
        thrust::device,
        dCellIdSorted,
        dCellIdSorted + n,
        dCellPoints
    );

    launchKernel(
        buildBinExtreme, smemForBuildBinExtreme, n,
        dCellStart, dCellEnd, dCellIdSorted
    );

    launchKernel(
        computeNeighborCount, nullptr, n,
        dNeighborCount, dX, dY, dCellId, dCellStart, dCellEnd, dCellPoints
    );

    CUDA_CHECK(cudaMemcpy(hNeighborCount, dNeighborCount, n * sizeof(int), cudaMemcpyDeviceToHost));

    int clusterId = NO_CLUSTER_LABEL;
    dim3 block(BLOCK_SIZE);

    for (int p = 0; p < n; p++) {
        if (!isCore(hNeighborCount[p], minPts)) continue;

        int hCluster;
        CUDA_CHECK(cudaMemcpy(&hCluster, dCluster + p, sizeof(int), cudaMemcpyDeviceToHost));
        if (hCluster != NO_CLUSTER_LABEL) continue;

        clusterId++;

        // TODO add this from the kernel in the loop
        // Assign cluster to core point
        CUDA_CHECK(cudaMemcpy(dCluster + p, &clusterId, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dFrontier, &p, sizeof(int), cudaMemcpyHostToDevice));

        int frontier_size = 1;
        while (frontier_size > 0) {
            CUDA_CHECK(cudaMemset(dNextFrontierSize, 0, sizeof(int)));

            dim3 bfs_grid((frontier_size + block.x - 1) / block.x);

            bfsExpand<<<bfs_grid, block>>>(
                dCluster, dNextFrontier, dNextFrontierSize,
                dX, dY, dCellStart, dCellEnd, dCellPoints,
                dNeighborCount, dFrontier, frontier_size, clusterId
            );

            CUDA_CHECK(cudaMemcpy(&frontier_size, dNextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost));
            std::swap(dFrontier, dNextFrontier);
        }
    }

    *clusterCount = clusterId;

    CUDA_CHECK(cudaMemcpy(cluster, dCluster, n * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dCluster));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dCellId));
    CUDA_CHECK(cudaFree(dCellIdSorted));
    CUDA_CHECK(cudaFree(dCellStart));
    CUDA_CHECK(cudaFree(dCellEnd));
    CUDA_CHECK(cudaFree(dCellPoints));
    CUDA_CHECK(cudaFree(dNeighborCount));
    CUDA_CHECK(cudaFree(dFrontier));
    CUDA_CHECK(cudaFree(dNextFrontier));
    CUDA_CHECK(cudaFree(dNextFrontierSize));
}
