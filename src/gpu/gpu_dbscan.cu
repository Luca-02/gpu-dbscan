#include <thrust/sort.h>
#include "./gpu_dbscan.h"
#include "../common.h"
#include "../helper.h"
#include "../cuda_helper.h"

__constant__ float cXMin;
__constant__ float cYMin;
__constant__ uint32_t cGridWidth;
__constant__ uint32_t cGridHeight;
__constant__ uint32_t cCellCount;
__constant__ uint32_t cN;
__constant__ float cEps;
__constant__ float cInvEps;
__constant__ float cEps2;
__constant__ uint32_t cMinPts;

size_t smemFindNextCore(const uint32_t blockSize) {
    return (blockSize + WARP_SIZE - 1 / WARP_SIZE) * sizeof(int);
}

__global__ void computeBinning(
    uint32_t *cellId,
    uint32_t *cellPoints,
    const float *x,
    const float *y
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < cN; i += stride) {
        uint32_t cx, cy;
        pointCellCoordinates(
            &cx, &cy,
            x[i], y[i],
            cXMin, cYMin, cInvEps
        );

        cellId[i] = linearCellId(cx, cy, cGridWidth);
        cellPoints[i] = i;
    }
}

__global__ void buildBinExtreme(
    uint32_t *cellStart,
    uint32_t *cellEnd,
    const uint32_t *cellId
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < cN; i += stride) {
        const uint32_t lane = threadIdx.x & (warpSize - 1); // threadIdx.x % 32
        const uint32_t cid = cellId[i];

        uint32_t prevCid = __shfl_up_sync(0xffffffff, cid, 1, warpSize);
        if (lane == 0 && i > 0) {
            prevCid = cellId[i - 1];
        }

        uint32_t nextCid = __shfl_down_sync(0xffffffff, cid, 1, warpSize);
        if (lane == warpSize - 1 && i + 1 < cN) {
            nextCid = cellId[i + 1];
        }

        if (i == 0 || cid != prevCid) {
            cellStart[cid] = i;
        }
        if (i == cN - 1 || cid != nextCid) {
            cellEnd[cid] = i + 1;
        }
    }
}

__global__ void computeIsCoreArr(
    uint8_t *isCoreArr,
    const float *x,
    const float *y,
    const uint32_t *cellStart,
    const uint32_t *cellEnd,
    const uint32_t *cellPoints
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < cN; i += stride) {
        const float xi = x[i];
        const float yi = y[i];

        uint32_t cx, cy;
        pointCellCoordinates(
            &cx, &cy,
            x[i], y[i],
            cXMin, cYMin, cInvEps
        );

        uint32_t count = 0;

        #pragma unroll
        for (int idx = 0; idx < 9; ++idx) {
            const int dx = idx / 3 - 1;
            const int dy = idx % 3 - 1;

            // Mask which is 0 if the cell is outside the border, 1 otherwise
            uint8_t inBounds = 1;
            inBounds &= !(dx < 0 && cx < (uint32_t) -dx);
            inBounds &= !(dy < 0 && cy < (uint32_t) -dy);
            inBounds &= cx + dx < cGridWidth;
            inBounds &= cy + dy < cGridHeight;

            const uint32_t nx = cx + dx;
            const uint32_t ny = cy + dy;

            const uint32_t nid = linearCellId(nx, ny, cGridWidth);
            const uint32_t start = inBounds ? cellStart[nid] : 0;
            const uint32_t end = inBounds ? cellEnd[nid] : 0;

            for (uint32_t k = start; k < end; k++) {
                const uint32_t j = cellPoints[k];
                const uint8_t valid = i != j && isEpsNeighbor(xi, yi, x[j], y[j], cEps2);
                count += valid;
            }
        }

        isCoreArr[i] = isCore(count, cMinPts);
    }
}

__global__ void findNextCore(
    uint32_t *nextCore,
    const uint32_t *cluster,
    const uint8_t *isCoreArr,
    const uint32_t startIdx
) {
    extern __shared__ uint32_t warpMins[];

    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    const uint32_t lane = threadIdx.x & (warpSize - 1); // threadIdx.x % 32
    const uint32_t wid = threadIdx.x >> 5; // threadIdx.x / 32

    uint32_t localMin = UINT_MAX;

    for (uint32_t i = tid + startIdx; i < cN; i += stride) {
        if (isCoreArr[i] && cluster[i] == NO_CLUSTER_LABEL) {
            localMin = min(localMin, i);
            break;
        }
    }

    // Reduce minimum within warp
    #pragma unroll
    for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
        const uint32_t other = __shfl_down_sync(0xffffffff, localMin, offset);
        localMin = min(localMin, other);
    }

    // Store each warp minimum
    if (lane == 0) {
        warpMins[wid] = localMin;
    }

    __syncthreads();

    // Reduce warp minimums within warp 0
    if (wid == 0) {
        uint32_t warpMin = lane < blockDim.x >> 5 ? warpMins[lane] : UINT_MAX;

        #pragma unroll
        for (uint32_t offset = warpSize / 2; offset > 0; offset >>= 1) {
            const uint32_t other = __shfl_down_sync(0xffffffff, warpMin, offset);
            warpMin = min(warpMin, other);
        }

        if (lane == 0 && warpMin != cN) {
            atomicMin(nextCore, warpMin);
        }
    }
}

__global__ void bfsExpand(
    uint32_t *cluster,
    uint32_t *frontier,
    uint32_t *nextFrontier,
    uint32_t *nextFrontierSize,
    const float *x,
    const float *y,
    const uint32_t *cellStart,
    const uint32_t *cellEnd,
    const uint32_t *cellPoints,
    const uint8_t *isCoreArr,
    const uint32_t frontierSize,
    const uint32_t clusterId,
    const uint32_t level,
    const uint32_t root
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    // Assign cluster to core point
    if (tid == 0 && level == 0) {
        cluster[root] = clusterId;
        frontier[0] = root;
    }

    __syncthreads();

    for (uint32_t f = tid; f < frontierSize; f += stride) {
        const uint32_t i = frontier[f];
        const float xi = x[i];
        const float yi = y[i];

        uint32_t cx, cy;
        pointCellCoordinates(
            &cx, &cy,
            xi, yi,
            cXMin, cYMin, cInvEps
        );

        #pragma unroll
        for (int idx = 0; idx < 9; ++idx) {
            const int dx = idx / 3 - 1;
            const int dy = idx % 3 - 1;

            // Mask which is 0 if the cell is outside the border, 1 otherwise
            uint8_t inBounds = 1;
            inBounds &= !(dx < 0 && cx < (uint32_t) -dx);
            inBounds &= !(dy < 0 && cy < (uint32_t) -dy);
            inBounds &= cx + dx < cGridWidth;
            inBounds &= cy + dy < cGridHeight;

            const uint32_t nx = cx + dx;
            const uint32_t ny = cy + dy;

            const uint32_t nid = linearCellId(nx, ny, cGridWidth);
            const uint32_t start = inBounds ? cellStart[nid] : 0;
            const uint32_t end = inBounds ? cellEnd[nid] : 0;

            for (uint32_t k = start; k < end; k++) {
                const uint32_t j = cellPoints[k];
                if (i == j || !isEpsNeighbor(xi, yi, x[j], y[j], cEps2)) continue;

                const uint32_t old = atomicCAS(&cluster[j], NO_CLUSTER_LABEL, clusterId);

                // Add neighbor to frontier iff it is not yet assigned to a cluster and is a core point
                if (old == NO_CLUSTER_LABEL && isCoreArr[j]) {
                    const uint32_t pos = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[pos] = j;
                }
            }
        }
    }
}

void dbscan_gpu(
    uint32_t *cluster,
    uint32_t *clusterCount,
    const float *x,
    const float *y,
    const uint32_t n,
    const float eps,
    const uint32_t minPts
) {
    // int device;
    // cudaDeviceProp prop;
    // cudaGetDevice(&device);
    // cudaGetDeviceProperties(&prop, device);

    float xMin = x[0], xMax = x[0];
    float yMin = y[0], yMax = y[0];
    for (uint32_t i = 1; i < n; i++) {
        xMin = fmin(xMin, x[i]);
        xMax = fmax(xMax, x[i]);
        yMin = fmin(yMin, y[i]);
        yMax = fmax(yMax, y[i]);
    }

    const uint32_t gridWidth = ceil((xMax - xMin) / eps + 1);
    const uint32_t gridHeight = ceil((yMax - yMin) / eps + 1);
    const uint32_t cellCount = gridWidth * gridHeight;
    const float invEps = 1.0 / eps;
    const float eps2 = eps * eps;

    uint32_t *dCluster;
    float *dX, *dY;
    uint32_t *dCellId;
    uint32_t *dCellStart;
    uint32_t *dCellEnd;
    uint32_t *dCellPoints;
    uint8_t *dIsCoreArr;
    uint32_t *dFrontier;
    uint32_t *dNextFrontier;
    uint32_t *dNextCore;
    uint32_t *dNextFrontierSize;

    CUDA_CHECK(cudaMalloc(&dCluster, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dX, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCellId, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dCellStart, cellCount * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dCellEnd, cellCount * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dCellPoints, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dIsCoreArr, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&dFrontier, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dNextFrontier, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dNextCore, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dNextFrontierSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpyToSymbol(cXMin, &xMin, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cYMin, &yMin, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cGridWidth, &gridWidth, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(cGridHeight, &gridHeight, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(cCellCount, &cellCount, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(cN, &n, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(cEps, &eps, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cInvEps, &invEps, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cEps2, &eps2, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cMinPts, &minPts, sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(dX, x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, y, n * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(dCluster, NO_CLUSTER_LABEL, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(dCellStart, 0, cellCount * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(dCellEnd, 0, cellCount * sizeof(uint32_t)));

    launchKernel(
        computeBinning, n,
        dCellId, dCellPoints, dX, dY
    );

    thrust::sort_by_key(
        thrust::device,
        dCellId,
        dCellId + n,
        dCellPoints
    );

    launchKernel(
        buildBinExtreme, n,
        dCellStart, dCellEnd, dCellId
    );

    launchKernel(
        computeIsCoreArr, n,
        dIsCoreArr, dX, dY, dCellStart, dCellEnd, dCellPoints
    );

    uint32_t startIdx = 0;
    uint32_t hClusterCount = NO_CLUSTER_LABEL;
    for (uint32_t iter = 0; iter < n; iter++) {
        uint32_t hNextCore = n;
        CUDA_CHECK(cudaMemcpy(dNextCore, &hNextCore,sizeof(uint32_t), cudaMemcpyHostToDevice));

        launchKernel(
            findNextCore, smemFindNextCore, n - startIdx,
            dNextCore, dCluster, dIsCoreArr, startIdx
        );

        CUDA_CHECK(cudaMemcpy(&hNextCore, dNextCore, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // If no more core points exit loop
        if (hNextCore >= n) break;

        startIdx = hNextCore + 1;
        hClusterCount++;

        uint32_t level = 0;
        uint32_t frontier_size = 1;
        while (frontier_size > 0) {
            CUDA_CHECK(cudaMemset(dNextFrontierSize, 0, sizeof(uint32_t)));

            launchKernel(
                bfsExpand, frontier_size,
                dCluster, dFrontier, dNextFrontier, dNextFrontierSize,
                dX, dY, dCellStart, dCellEnd, dCellPoints, dIsCoreArr,
                frontier_size, hClusterCount, level, hNextCore
            );

            level++;
            CUDA_CHECK(cudaMemcpy(&frontier_size, dNextFrontierSize, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            std::swap(dFrontier, dNextFrontier);
        }
    }

    *clusterCount = hClusterCount;

    CUDA_CHECK(cudaMemcpy(cluster, dCluster, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

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
