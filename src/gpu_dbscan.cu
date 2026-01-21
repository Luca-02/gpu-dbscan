#include <cccl/thrust/sort.h>
#include "dbscan.h"
#include "common.h"

#define BLOCK_SIZE 1024

__global__ void init_point_idx(int *point_idx, const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        point_idx[idx] = idx;
    }
}

__global__ void compute_cell_id(
    int *cell_id,
    const double *x,
    const double *y,
    const double x_min,
    const double y_min,
    const int grid_width,
    const int n,
    const double eps
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int cx, cy;
    cell_coordinates(&cx, &cy, x[tid], y[tid], x_min, y_min, eps);
    cell_id[tid] = cx + cy * grid_width;
}

__global__ void build_cell_offset(
    int *cell_start,
    int *cell_end,
    const int *cell_id,
    const int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const int cid = cell_id[tid];

    if (tid == 0 || cid != cell_id[tid - 1]) {
        cell_start[cid] = tid;
    }

    if (tid == n - 1 || cid != cell_id[tid + 1]) {
        cell_end[cid] = tid + 1;
    }
}

__global__ void compute_neighbor_count(
    int *neighbor_count,
    const double *x,
    const double *y,
    const int n,
    const double eps
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const double xi = x[tid];
    const double yi = y[tid];

    int count = 0;
    for (int j = 0; j < n; j++) {
        if (is_eps_neighbor(xi, yi, x[j], y[j], eps)) {
            count++;
        }
    }

    neighbor_count[tid] = count;
}

__global__ void compute_neighbor_count(
    int *neighbor_count,
    const double *x,
    const double *y,
    const int *cell_id,
    const int *cell_start,
    const int *cell_end,
    const int *point_idx,
    const int grid_width,
    const int grid_height,
    const int n,
    const double eps
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const int i = point_idx[tid];
    const double xi = x[i];
    const double yi = y[i];
    const int cid = cell_id[tid];

    const int cx = cid % grid_width;
    const int cy = cid / grid_width;

    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= grid_width || ny >= grid_height) continue;

            const int nid = ny * grid_width + nx;
            const int start = cell_start[nid];
            const int end   = cell_end[nid];

            for (int k = start; k < end; k++) {
                const int j = point_idx[k];
                if (is_eps_neighbor(xi, yi, x[j], y[j], eps))
                    count++;
            }
        }
    }

    neighbor_count[i] = count;
}

__global__ void bfs_expand(
    int *cluster,
    int *next_frontier,
    int *next_frontier_size,
    const double *x,
    const double *y,
    const int *neighbor_count,
    const int *frontier,
    const int frontier_size,
    const int cluster_id,
    const int n,
    const double eps,
    const int min_pts
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    const int i = frontier[tid];
    const double xi = x[i];
    const double yi = y[i];

    for (int j = 0; j < n; j++) {
        if (i == j) continue;

        if (is_eps_neighbor(xi, yi, x[j], y[j], eps)) {
            const int old = atomicCAS(&cluster[j], NO_CLUSTER, cluster_id);

            if (old == NO_CLUSTER && is_core(neighbor_count[j], min_pts)) {
                const int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = j;
            }
        }
    }
}

__global__ void bfs_expand_opt(
    int *cluster,
    int *next_frontier,
    int *next_frontier_size,
    const double *x,
    const double *y,
    const int *neighbor_count,
    const int *frontier,
    const int frontier_size,
    const int cluster_id,
    const int n,
    const double eps,
    const int min_pts,
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

        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            if (is_eps_neighbor(xi, yi, x[j], y[j], eps)) {
                const int old = atomicCAS(&cluster[j], NO_CLUSTER, cluster_id);

                if (old == NO_CLUSTER && is_core(neighbor_count[j], min_pts)) {
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

__global__ void bfs_expand_grid(
    int *cluster,
    int *next_frontier,
    int *next_frontier_size,
    const double *x,
    const double *y,
    const int *cell_start,
    const int *cell_end,
    const int *point_idx,
    const int grid_width,
    const int grid_height,
    const double x_min,
    const double y_min,
    const int *neighbor_count,
    const int *frontier,
    const int frontier_size,
    const int cluster_id,
    const double eps,
    const int min_pts
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    const int i = frontier[tid];
    const double xi = x[i];
    const double yi = y[i];

    int cx, cy;
    cell_coordinates(&cx, &cy, xi, yi, x_min, y_min, eps);

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= grid_width || ny >= grid_height) continue;

            const int cid = nx + ny * grid_width;
            const int start = cell_start[cid];
            const int end = cell_end[cid];

            for (int k = start; k < end; k++) {
                const int j = point_idx[k];
                if (i == j) continue;

                if (is_eps_neighbor(xi, yi, x[j], y[j], eps)) {
                    const int old = atomicCAS(&cluster[j], NO_CLUSTER, cluster_id);

                    if (old == NO_CLUSTER && is_core(neighbor_count[j], min_pts)) {
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
    const double *x,
    const double *y,
    const int n,
    const double eps,
    const int min_pts
) {
    int *h_neighbor_count = (int *) malloc_s(n * sizeof(int));
    if (!h_neighbor_count) return;

    int *d_cluster;
    double *d_x, *d_y;
    int *d_neighbor_count;
    int *d_frontier, *d_next_frontier;
    int *d_next_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_cluster, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighbor_count, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_cluster, NO_CLUSTER, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x);

    compute_neighbor_count<<<grid, block>>>(d_neighbor_count, d_x, d_y, n, eps);

    CUDA_CHECK(cudaMemcpy(h_neighbor_count, d_neighbor_count, n * sizeof(int), cudaMemcpyDeviceToHost));

    int cluster_id = NO_CLUSTER;

    for (int p = 0; p < n; p++) {
        if (!is_core(h_neighbor_count[p], min_pts)) continue;

        int h_cluster;
        CUDA_CHECK(cudaMemcpy(&h_cluster, d_cluster + p, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_cluster != NO_CLUSTER) continue;

        cluster_id++;

        // Assign cluster to core point
        CUDA_CHECK(cudaMemcpy(d_cluster + p, &cluster_id, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_frontier, &p, sizeof(int), cudaMemcpyHostToDevice));

        int frontier_size = 1;
        while (frontier_size > 0) {
            CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

            dim3 grid_frontier((frontier_size + block.x - 1) / block.x);

            bfs_expand<<<grid_frontier, block>>>(
                d_cluster,
                d_next_frontier,
                d_next_frontier_size,
                d_x,
                d_y,
                d_neighbor_count,
                d_frontier,
                frontier_size,
                cluster_id,
                n,
                eps,
                min_pts
            );

            CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
            std::swap(d_frontier, d_next_frontier);
        }
    }

    *cluster_count = cluster_id;

    CUDA_CHECK(cudaMemcpy(cluster, d_cluster, n * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_cluster));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_neighbor_count));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier_size));

    cudaDeviceReset();
}

void dbscan_gpu_opt(
    int *cluster,
    int *cluster_count,
    const double *x,
    const double *y,
    const int n,
    const double eps,
    const int min_pts
) {
    int *h_neighbor_count = (int *) malloc_s(n * sizeof(int));
    if (!h_neighbor_count) return;

    int *d_cluster;
    double *d_x, *d_y;
    int *d_neighbor_count;
    int *d_frontier, *d_next_frontier;
    int *d_next_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_cluster, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighbor_count, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_cluster, NO_CLUSTER, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x);

    compute_neighbor_count<<<grid, block>>>(d_neighbor_count, d_x, d_y, n, eps);

    CUDA_CHECK(cudaMemcpy(h_neighbor_count, d_neighbor_count, n * sizeof(int), cudaMemcpyDeviceToHost));

    const int smem_frontier_size = block.x;
    const size_t dynamic_smem = smem_frontier_size * sizeof(int) + sizeof(int);
    int cluster_id = NO_CLUSTER;

    for (int p = 0; p < n; p++) {
        if (!is_core(h_neighbor_count[p], min_pts)) continue;

        int h_cluster;
        CUDA_CHECK(cudaMemcpy(&h_cluster, d_cluster + p, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_cluster != NO_CLUSTER) continue;

        cluster_id++;

        // Assign cluster to core point
        CUDA_CHECK(cudaMemcpy(d_cluster + p, &cluster_id, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_frontier, &p, sizeof(int), cudaMemcpyHostToDevice));

        int frontier_size = 1;
        while (frontier_size > 0) {
            CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

            dim3 grid_frontier((frontier_size + block.x - 1) / block.x);

            bfs_expand_opt<<<grid_frontier, block, dynamic_smem>>>(
                d_cluster,
                d_next_frontier,
                d_next_frontier_size,
                d_x,
                d_y,
                d_neighbor_count,
                d_frontier,
                frontier_size,
                cluster_id,
                n,
                eps,
                min_pts,
                smem_frontier_size
            );

            CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
            std::swap(d_frontier, d_next_frontier);
        }
    }

    *cluster_count = cluster_id;

    CUDA_CHECK(cudaMemcpy(cluster, d_cluster, n * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_cluster));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_neighbor_count));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier_size));

    cudaDeviceReset();
}

void dbscan_gpu_grid(
    int *cluster,
    int *cluster_count,
    const double *x,
    const double *y,
    const int n,
    const double eps,
    const int min_pts
) {
    double x_min = x[0], x_max = x[0], y_min = y[0], y_max = y[0];
    for (int i = 1; i < n; i++) {
        x_min = min(x_min, x[i]);
        x_max = max(x_max, x[i]);
        y_min = min(y_min, y[i]);
        y_max = max(y_max, y[i]);
    }

    const int grid_width = ceil((x_max - x_min) / eps) + 1;
    const int grid_height = ceil((y_max - y_min) / eps) + 1;
    const int cell_count = grid_width * grid_height;

    int *h_neighbor_count = (int *) malloc_s(n * sizeof(int));
    if (!h_neighbor_count) return;

    int *d_cluster;
    double *d_x, *d_y;
    int *d_cell_id;
    int *d_cell_start, *d_cell_end;
    int *d_point_idx;
    int *d_neighbor_count;
    int *d_frontier, *d_next_frontier;
    int *d_next_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_cluster, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cell_id, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_start, cell_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cell_end, cell_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_point_idx, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighbor_count, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(int)));

    CUDA_CHECK(cudaMemset(d_cluster, NO_CLUSTER, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x);

    init_point_idx<<<grid, block>>>(d_point_idx, n);

    compute_cell_id<<<grid, block>>>(d_cell_id, d_x, d_y, x_min, y_min, grid_width, n, eps);

    thrust::sort_by_key(thrust::device, d_cell_id, d_cell_id + n, d_point_idx);

    build_cell_offset<<<grid, block>>>(d_cell_start, d_cell_end, d_cell_id, n);

    compute_neighbor_count<<<grid, block>>>(
        d_neighbor_count, d_x, d_y, d_cell_id, d_cell_start,
        d_cell_end, d_point_idx, grid_width, grid_height, n, eps
    );

    CUDA_CHECK(cudaMemcpy(h_neighbor_count, d_neighbor_count, n * sizeof(int), cudaMemcpyDeviceToHost));

    int cluster_id = NO_CLUSTER;

    for (int p = 0; p < n; p++) {
        if (!is_core(h_neighbor_count[p], min_pts)) continue;

        int h_cluster;
        CUDA_CHECK(cudaMemcpy(&h_cluster, d_cluster + p, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_cluster != NO_CLUSTER) continue;

        cluster_id++;

        // Assign cluster to core point
        CUDA_CHECK(cudaMemcpy(d_cluster + p, &cluster_id, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_frontier, &p, sizeof(int), cudaMemcpyHostToDevice));

        int frontier_size = 1;
        while (frontier_size > 0) {
            CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(int)));

            dim3 grid_frontier((frontier_size + block.x - 1) / block.x);

            bfs_expand_grid<<<grid_frontier, block>>>(
                d_cluster, d_next_frontier, d_next_frontier_size, d_x, d_y,
                d_cell_start, d_cell_end, d_point_idx, grid_width, grid_height,
                x_min, y_min, d_neighbor_count, d_frontier, frontier_size,
                cluster_id, eps, min_pts
            );

            CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_frontier_size, sizeof(int), cudaMemcpyDeviceToHost));
            std::swap(d_frontier, d_next_frontier);
        }
    }

    *cluster_count = cluster_id;

    CUDA_CHECK(cudaMemcpy(cluster, d_cluster, n * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_cluster));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_cell_id));
    CUDA_CHECK(cudaFree(d_cell_start));
    CUDA_CHECK(cudaFree(d_cell_end));
    CUDA_CHECK(cudaFree(d_point_idx));
    CUDA_CHECK(cudaFree(d_neighbor_count));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier_size));

    cudaDeviceReset();
}
