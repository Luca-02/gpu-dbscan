#include <thrust/sort.h>
#include "gpu_dbscan.h"
#include "common.h"
#include "helper.h"
#include "cuda_helper.h"

#define BLOCK_SIZE 512
#define BFS_BLOCK_SIZE 512

__constant__ double c_x_min;
__constant__ double c_y_min;
__constant__ int c_grid_width;
__constant__ int c_grid_height;
__constant__ int c_cell_count;
__constant__ int c_n;
__constant__ double c_eps;
__constant__ double c_inv_eps;
__constant__ double c_eps2;
__constant__ int c_min_pts;

__global__ void compute_binning(
    int *cell_id,
    int *cell_points,
    const double *x,
    const double *y
) {
    extern __shared__ double s_x[];
    extern __shared__ double s_y[];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < c_n) {
        s_x[threadIdx.x] = x[tid];
        s_y[threadIdx.x] = y[tid];
    }

    __syncthreads();

    if (tid < c_n) {
        int cx, cy;
        pointCellCoordinates(
            &cx, &cy,
            s_x[threadIdx.x], s_y[threadIdx.x],
            c_x_min, c_y_min, c_inv_eps
        );

        cell_id[tid] = linearCellId(cx, cy, c_grid_width);
        cell_points[tid] = tid;
    }
}

__global__ void built_bin_extremes(
    int *cell_start,
    int *cell_end,
    const int *cell_id_sorted
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= c_n) return;

    const int cid = cell_id_sorted[tid];

    if (tid == 0 || cid != cell_id_sorted[tid - 1]) {
        cell_start[cid] = tid;
    }

    if (tid == c_n - 1 || cid != cell_id_sorted[tid + 1]) {
        cell_end[cid] = tid + 1;
    }
}

__global__ void built_bin_extremes_smem(
    int *cell_start,
    int *cell_end,
    const int *cell_id_sorted
) {
    extern __shared__ int smem[];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x;

    if (lane == 0) {
        smem[0] = tid > 0 ? cell_id_sorted[tid - 1] : -1;
    }

    if (tid < c_n) {
        smem[lane + 1] = cell_id_sorted[tid];
    }

    if (lane == blockDim.x - 1) {
        smem[blockDim.x + 1] = tid + 1 < c_n ? cell_id_sorted[tid + 1] : -1;
    }

    __syncthreads();

    if (tid >= c_n) return;

    const int prev_cid = smem[lane];
    const int cid = smem[lane + 1];
    const int next_cid = smem[lane + 2];

    if (tid == 0 || cid != prev_cid) {
        cell_start[cid] = tid;
    }

    if (tid == c_n - 1 || cid != next_cid) {
        cell_end[cid] = tid + 1;
    }
}

__global__ void build_bin_extreme_opt(
    int *cell_start,
    int *cell_end,
    const int *cell_id_sorted
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warpSize;
    if (tid >= c_n) return;

    const int cid = cell_id_sorted[tid];

    int prev_cid = __shfl_up_sync(0xffffffff, cid, 1, warpSize);
    // The first thread in the warp need to get the previous cid directly from the global memory
    if (lane == 0 && tid > 0) {
        prev_cid = cell_id_sorted[tid - 1];
    }

    int next_cid = __shfl_down_sync(0xffffffff, cid, 1, warpSize);
    // The last thread in the warp need to get the next cid directly from the global memory
    if (lane == warpSize - 1 && tid + 1 < c_n) {
        next_cid = cell_id_sorted[tid + 1];
    }

    if (tid == 0 || cid != prev_cid) {
        cell_start[cid] = tid;
    }
    if (tid == c_n - 1 || cid != next_cid) {
        cell_end[cid] = tid + 1;
    }
}

__global__ void build_bin_end(
    int *cell_end,
    const int *cell_start
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= c_cell_count) return;

    // Find the next valid cell
    int next = tid + 1;
    while (next < c_cell_count && cell_start[next] == -1) {
        next++;
    }

    cell_end[tid] = next < c_cell_count ? cell_start[next] : c_n;
}

__global__ void neighbor_counts(
    int *neighbor_counts,
    const double *x,
    const double *y,
    const int *cell_id,
    const int *cell_start,
    const int *cell_end,
    const int *cell_points
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= c_n) return;

    const double xi = x[tid];
    const double yi = y[tid];

    const int cid = cell_id[tid];
    const int cx = cid % c_grid_width;
    const int cy = cid / c_grid_width;

    int count = 0;

#pragma unroll
    for (int dx = -1; dx <= 1; dx++) {
#pragma unroll
        for (int dy = -1; dy <= 1; dy++) {
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= c_grid_width || ny >= c_grid_height) continue;

            const int nid = linearCellId(nx, ny, c_grid_width);
            const int start = cell_start[nid];
            const int end = cell_end[nid];

            for (int k = start; k < end; k++) {
                const int j = cell_points[k];

                if (tid != j && isEpsNeighbor(xi, yi, x[j], y[j], c_eps2)) {
                    count++;
                }
            }
        }
    }

    neighbor_counts[tid] = count;
}

__global__ void bfs_expand(
    int *cluster,
    int *next_frontier,
    int *next_frontier_size,
    const double *x,
    const double *y,
    const int *cell_start,
    const int *cell_end,
    const int *cell_points,
    const int *neighbor_count,
    const int *frontier,
    const int frontier_size,
    const int cluster_id
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    const int i = frontier[tid];
    const double xi = x[i];
    const double yi = y[i];

    int cx, cy;
    pointCellCoordinates(
        &cx, &cy,
        xi, yi,
        c_x_min, c_y_min, c_inv_eps
    );

#pragma unroll
    for (int dx = -1; dx <= 1; dx++) {
#pragma unroll
        for (int dy = -1; dy <= 1; dy++) {
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= c_grid_width || ny >= c_grid_height) continue;

            const int nid = linearCellId(nx, ny, c_grid_width);
            const int start = cell_start[nid];
            const int end = cell_end[nid];

            for (int k = start; k < end; k++) {
                const int j = cell_points[k];

                if (i != j && isEpsNeighbor(xi, yi, x[j], y[j], c_eps2)) {
                    const int old = atomicCAS(&cluster[j], NO_CLUSTER_LABEL, cluster_id);

                    // Add neighbor to frontier iff it is not yet assigned to a cluster and is a core point
                    if (old == NO_CLUSTER_LABEL && isCore(neighbor_count[j], c_min_pts)) {
                        const int pos = atomicAdd(next_frontier_size, 1);
                        next_frontier[pos] = j;
                    }
                }
            }
        }
    }
}

size_t sharedMemForBinning(const int blockSize) {
    return blockSize * 2 * sizeof(double);
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
    int device;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    double x_min = x[0], x_max = x[0];
    double y_min = y[0], y_max = y[0];
    for (int i = 1; i < n; i++) {
        x_min = fmin(x_min, x[i]);
        x_max = fmax(x_max, x[i]);
        y_min = fmin(y_min, y[i]);
        y_max = fmax(y_max, y[i]);
    }

    const int grid_width = ceil((x_max - x_min) / eps + 1);
    const int grid_height = ceil((y_max - y_min) / eps + 1);
    const int cell_count = grid_width * grid_height;
    const double inv_eps = 1.0 / eps;
    const double eps2 = eps * eps;

    int *h_neighbor_counts = (int *) malloc_s(n * sizeof(int));
    if (!h_neighbor_counts) return;

    int *d_cluster;
    double *d_x, *d_y;
    int *d_cell_id;
    int *d_cell_id_sorted;
    int *d_cell_start;
    int *d_cell_end;
    int *d_cell_points;
    int *d_neighbor_count;
    int *d_frontier;
    int *d_next_frontier;
    int *d_next_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_cluster, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cell_id, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_id_sorted, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_start, cell_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_end, cell_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_points, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighbor_count, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(int)));

    CUDA_CHECK(cudaMemcpyToSymbol(c_x_min, &x_min, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_y_min, &y_min, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_grid_width, &grid_width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_grid_height, &grid_height, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_cell_count, &cell_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_n, &n, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_eps, &eps, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_inv_eps, &inv_eps, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_eps2, &eps2, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_min_pts, &minPts, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_cluster, NO_CLUSTER_LABEL, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cell_start, 0, cell_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cell_end, 0, cell_count * sizeof(int)));

    launchKernel(
        &prop, true, compute_binning, "compute_binning", n, sharedMemForBinning,
        d_cell_id, d_cell_points, d_x, d_y
    );

    // CUDA_CHECK(cudaMemcpy(d_cell_id_sorted, d_cell_id, n * sizeof(int), cudaMemcpyDeviceToDevice));
    //
    // thrust::sort_by_key(
    //     thrust::device,
    //     d_cell_id_sorted,
    //     d_cell_id_sorted + n,
    //     d_cell_points
    // );
    //
    // cudaEvent_t start, stop;
    // float elapsedTime;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    //
    // cudaEventRecord(start);
    // launchKernel(
    //     &prop, true, built_bin_extremes, "built_bin_extremes", n, 0,
    //     d_cell_start, d_cell_end, d_cell_id_sorted
    // );
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf("Time to exec built_bin_extremes (old): %f ms\n", elapsedTime);
    //
    // cudaEventRecord(start);
    // launchKernel(
    //     &prop, true, built_bin_extremes_smem, "built_bin_extremes_smem", n, sharedMemBytes,
    //     d_cell_start, d_cell_end, d_cell_id_sorted
    // );
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf("Time to exec built_bin_extremes_smem (smem): %f ms\n", elapsedTime);
    //
    // cudaEventRecord(start);
    // build_bin_extreme_opt<<<grid, block>>>(d_cell_start, d_cell_end, d_cell_id_sorted);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf("Time to exec build_bin_extreme_opt (warp): %f ms\n", elapsedTime);

    // neighbor_counts<<<grid, block>>>(
    //     d_neighbor_count, d_x, d_y, d_cell_id,
    //     d_cell_start, d_cell_end, d_cell_points
    // );
    //
    // CUDA_CHECK(cudaMemcpy(h_neighbor_counts, d_neighbor_count, n * sizeof(int), cudaMemcpyDeviceToHost));

    // int cluster_id = NO_CLUSTER_LABEL;
    // dim3 bfs_block(BFS_BLOCK_SIZE);
    //
    // for (int p = 0; p < n; p++) {
    //     if (!isCore(h_neighbor_counts[p], minPts)) continue;
    //
    //     int h_cluster;
    //     CUDA_CHECK(cudaMemcpy(&h_cluster, d_cluster + p, sizeof(int), cudaMemcpyDeviceToHost));
    //     if (h_cluster != NO_CLUSTER_LABEL) continue;
    //
    //     cluster_id++;
    //
    //     // TODO add this from the kernel in the loop
    //     // Assign cluster to core point
    //     CUDA_CHECK(cudaMemcpy(d_cluster + p, &cluster_id, sizeof(int), cudaMemcpyHostToDevice));
    //     CUDA_CHECK(cudaMemcpy(d_frontier, &p, sizeof(int), cudaMemcpyHostToDevice));
    //
    //     int frontier_size = 1;
    //     while (frontier_size > 0) {
    //         CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));
    //
    //         dim3 bfs_grid((frontier_size + bfs_block.x - 1) / bfs_block.x);
    //
    //         bfs_expand<<<bfs_grid, bfs_block>>>(
    //             d_cluster, d_next_frontier, d_next_frontier_size,
    //             d_x, d_y, d_cell_start, d_cell_end, d_cell_points,
    //             d_neighbor_count, d_frontier, frontier_size, cluster_id
    //         );
    //
    //         CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
    //         std::swap(d_frontier, d_next_frontier);
    //     }
    // }
    //
    // *clusterCount = cluster_id;

    CUDA_CHECK(cudaMemcpy(cluster, d_cluster, n * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_cluster));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_cell_id));
    CUDA_CHECK(cudaFree(d_cell_id_sorted));
    CUDA_CHECK(cudaFree(d_cell_start));
    CUDA_CHECK(cudaFree(d_cell_end));
    CUDA_CHECK(cudaFree(d_cell_points));
    CUDA_CHECK(cudaFree(d_neighbor_count));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier_size));
}
