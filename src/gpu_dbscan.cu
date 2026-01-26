#include <cccl/thrust/sort.h>
#include "gpu_dbscan.h"
#include "common.h"
#include "helper.h"

#define BLOCK_SIZE 1024
#define BFS_BLOCK_SIZE 128

__constant__ double c_x_min;
__constant__ double c_y_min;
__constant__ int c_grid_width;
__constant__ int c_grid_height;
__constant__ int c_n;
__constant__ double c_eps;
__constant__ double c_eps2;
__constant__ int c_min_pts;

__global__ void binning(
    int *cell_ids,
    int *cell_points,
    const double *points
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= c_n) return;

    int cx, cy;
    point_cell_coordinates(
        &cx, &cy,
        points[X_INDEX(tid)], points[Y_INDEX(tid)],
        c_x_min, c_y_min, c_eps
    );

    cell_ids[tid] = cell_id(cx, cy, c_grid_width);
    cell_points[tid] = tid;
}

__global__ void bin_extremes(
    int *cell_starts,
    int *cell_offsets,
    const int *cell_ids
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= c_n) return;

    const int cid = cell_ids[tid];

    if (tid == 0 || cid != cell_ids[tid - 1]) {
        cell_starts[cid] = tid;
    }

    if (tid == c_n - 1 || cid != cell_ids[tid + 1]) {
        cell_offsets[cid] = tid + 1;
    }
}

__global__ void neighbor_counts(
    int *neighbor_counts,
    const double *points,
    const int *cell_ids,
    const int *cell_starts,
    const int *cell_offsets,
    const int *cell_points
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= c_n) return;

    const int i = cell_points[tid];
    const double xi = points[X_INDEX(i)];
    const double yi = points[Y_INDEX(i)];
    const int cid = cell_ids[tid];

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

            const int nid = cell_id(nx, ny, c_grid_width);
            const int start = cell_starts[nid];
            const int offset = cell_offsets[nid];
            if (start == -1) continue;

            for (int k = start; k < offset; k++) {
                const int j = cell_points[k];
                const double xj = points[X_INDEX(j)];
                const double yj = points[Y_INDEX(j)];

                if (i != j && is_eps_neighbor(xi, yi, xj, yj, c_eps2)) {
                    count++;
                }
            }
        }
    }

    neighbor_counts[i] = count;
}

__global__ void bfs_expand_opt(
    int *cluster,
    int *next_frontier,
    int *next_frontier_size,
    const double *x,
    const double *y,
    const int *neighbor_counts,
    const int *frontier,
    const int frontier_size,
    const int cluster_id,
    const int smem_size
) {
    extern __shared__ int smem[];

    int &smem_count = smem[0];
    int *smem_frontier = &smem[1];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        smem_count = 0;
    }

    __syncthreads();

    if (tid < frontier_size) {
        const int i = frontier[tid];
        const double xi = x[i];
        const double yi = y[i];

        for (int j = 0; j < c_n; j++) {
            if (i != j && is_eps_neighbor(xi, yi, x[j], y[j], c_eps2)) {
                const int old = atomicCAS(&cluster[j], NO_CLUSTER_LABEL, cluster_id);

                if (old == NO_CLUSTER_LABEL && is_core(neighbor_counts[j], c_min_pts)) {
                    const int smem_pos = atomicAdd(&smem_count, 1);

                    if (smem_pos < smem_size) {
                        smem_frontier[smem_pos] = j;
                    } else {
                        // Smem overflow protection, write directly to global memory
                        // Overflow occurs when the block has discovered more core
                        // neighbors than can be stored in its smem frontier.
                        const int global_pos = atomicAdd(next_frontier_size, 1);
                        next_frontier[global_pos] = j;
                    }
                }
            }
        }
    }

    __syncthreads();

    __shared__ int global_pos_block;

    if (threadIdx.x == 0 && smem_count > 0) {
        const int smem_count_limited = min(smem_count, smem_size);
        global_pos_block = atomicAdd(next_frontier_size, smem_count_limited);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < smem_count && i < smem_size; i += blockDim.x) {
        next_frontier[global_pos_block + i] = smem_frontier[i];
    }
}

// TODO check for two-phase BFS optimization (pro style)
__global__ void bfs_expand(
    int *cluster,
    int *next_frontier,
    int *next_frontier_size,
    const double *points,
    const int *cell_starts,
    const int *cell_offsets,
    const int *cell_points,
    const int *neighbor_counts,
    const int *frontier,
    const int frontier_size,
    const int cluster_id
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    const int i = frontier[tid];
    const double xi = points[X_INDEX(i)];
    const double yi = points[Y_INDEX(i)];

    int cx, cy;
    point_cell_coordinates(&cx, &cy, xi, yi, c_x_min, c_y_min, c_eps);

    #pragma unroll
    for (int dx = -1; dx <= 1; dx++) {
        #pragma unroll
        for (int dy = -1; dy <= 1; dy++) {
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= c_grid_width || ny >= c_grid_height) continue;

            const int nid = cell_id(nx, ny, c_grid_width);
            const int start = cell_starts[nid];
            const int offset = cell_offsets[nid];
            if (start == -1) continue;

            for (int k = start; k < offset; k++) {
                const int j = cell_points[k];
                const double xj = points[X_INDEX(j)];
                const double yj = points[Y_INDEX(j)];

                if (i != j && is_eps_neighbor(xi, yi, xj, yj, c_eps2)) {
                    const int old = atomicCAS(&cluster[j], NO_CLUSTER_LABEL, cluster_id);

                    if (old == NO_CLUSTER_LABEL && is_core(neighbor_counts[j], c_min_pts)) {
                        const int pos = atomicAdd(next_frontier_size, 1);
                        next_frontier[pos] = j;
                    }
                }
            }
        }
    }
}

void dbscan_gpu(
    int *cluster,
    int *cluster_count,
    const double *points,
    const int n,
    const double eps,
    const int min_pts
) {
    double x_min = points[X_INDEX(0)], x_max = points[X_INDEX(0)];
    double y_min = points[Y_INDEX(0)], y_max = points[Y_INDEX(0)];
    for (int i = 1; i < n; i++) {
        x_min = fmin(x_min, points[X_INDEX(i)]);
        x_max = fmax(x_max, points[X_INDEX(i)]);
        y_min = fmin(y_min, points[Y_INDEX(i)]);
        y_max = fmax(y_max, points[Y_INDEX(i)]);
    }

    const int grid_width = ceil((x_max - x_min) / eps + 1);
    const int grid_height = ceil((y_max - y_min) / eps + 1);
    const int cell_count = grid_width * grid_height;
    const double eps2 = eps * eps;

    int *h_neighbor_counts = (int *) malloc_s(n * sizeof(int));
    if (!h_neighbor_counts) return;

    int *d_cluster;
    double *d_points;
    int *d_cell_ids;
    int *d_cell_starts, *d_cell_offsets;
    int *d_cell_points;
    int *d_neighbor_counts;
    int *d_frontier, *d_next_frontier;
    int *d_next_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_cluster, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_points, n * 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cell_ids, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_starts, cell_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_offsets, cell_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_points, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighbor_counts, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(int)));

    CUDA_CHECK(cudaMemcpyToSymbol(c_x_min, &x_min, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_y_min, &y_min, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_grid_width, &grid_width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_grid_height, &grid_height, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_n, &n, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_eps, &eps, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_eps2, &eps2, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_min_pts, &min_pts, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points, points, n * 2 * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_cluster, NO_CLUSTER_LABEL, n * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cell_starts, -1, cell_count * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cell_offsets, -1, cell_count * sizeof(int)));

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x);

    binning<<<grid, block>>>(d_cell_ids, d_cell_points, d_points);

    thrust::sort_by_key(thrust::device, d_cell_ids, d_cell_ids + n, d_cell_points);

    CUDA_CHECK(cudaDeviceSynchronize());

    bin_extremes<<<grid, block>>>(d_cell_starts, d_cell_offsets, d_cell_ids);

    neighbor_counts<<<grid, block>>>(
        d_neighbor_counts, d_points, d_cell_ids,
        d_cell_starts, d_cell_offsets, d_cell_points
    );

    CUDA_CHECK(cudaMemcpy(h_neighbor_counts, d_neighbor_counts, n * sizeof(int), cudaMemcpyDeviceToHost));

    int cluster_id = NO_CLUSTER_LABEL;
    dim3 bfs_block(BFS_BLOCK_SIZE);

    for (int p = 0; p < n; p++) {
        if (!is_core(h_neighbor_counts[p], min_pts)) continue;

        int h_cluster;
        CUDA_CHECK(cudaMemcpy(&h_cluster, d_cluster + p, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_cluster != NO_CLUSTER_LABEL) continue;

        cluster_id++;

        // TODO add this from the kernel in the loop
        // Assign cluster to core point
        CUDA_CHECK(cudaMemcpy(d_cluster + p, &cluster_id, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_frontier, &p, sizeof(int), cudaMemcpyHostToDevice));

        int frontier_size = 1;
        while (frontier_size > 0) {
            CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

            dim3 bfs_grid((frontier_size + bfs_block.x - 1) / bfs_block.x);

            bfs_expand<<<bfs_grid, bfs_block>>>(
                d_cluster, d_next_frontier, d_next_frontier_size,
                d_points, d_cell_starts, d_cell_offsets,
                d_cell_points, d_neighbor_counts, d_frontier,
                frontier_size, cluster_id
            );

            CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
            std::swap(d_frontier, d_next_frontier);
        }
    }

    *cluster_count = cluster_id;

    CUDA_CHECK(cudaMemcpy(cluster, d_cluster, n * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_cluster));
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_cell_ids));
    CUDA_CHECK(cudaFree(d_cell_starts));
    CUDA_CHECK(cudaFree(d_cell_offsets));
    CUDA_CHECK(cudaFree(d_cell_points));
    CUDA_CHECK(cudaFree(d_neighbor_counts));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier_size));
}
