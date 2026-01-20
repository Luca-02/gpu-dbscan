#include "dbscan.h"
#include "helper.h"

#define BLOCK_SIZE 256

__global__ void init_cluster_kernel(int *cluster, const size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cluster[i] = NO_CLUSTER;
    }
}

__global__ void count_neighbors(
    size_t *neighbor_count,
    bool *core,
    const double *x,
    const double *y,
    const size_t n,
    const double eps,
    const size_t min_pts
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double xi = x[i];
    const double yi = y[i];

    size_t count = 0;
    for (size_t j = 0; j < n; j++) {
        if (is_eps_neighbor(xi, yi, x[j], y[j], eps)) {
            count++;
        }
    }

    neighbor_count[i] = count;
    core[i] = is_core(count, min_pts);
}

__global__ void bfs_expand(
    int *cluster,
    size_t *next_frontier,
    size_t *next_frontier_size,
    const double *x,
    const double *y,
    const size_t *neighbor_count,
    const size_t *frontier,
    const size_t frontier_size,
    const int cluster_id,
    const size_t n,
    const double eps,
    const size_t min_pts
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    const size_t q = frontier[idx];
    const double xq = x[q];
    const double yq = y[q];

    for (size_t k = 0; k < n; k++) {
        if (q == k) continue;

        if (is_eps_neighbor(xq, yq, x[k], y[k], eps)) {
            const int old = atomicCAS(&cluster[k], NO_CLUSTER, cluster_id);

            if (old == NO_CLUSTER && is_core(neighbor_count[k], min_pts)) {
                const size_t pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = k;
            }
        }
    }
}

void dbscan_gpu(
    int *cluster,
    size_t *cluster_count,
    const double *x,
    const double *y,
    const size_t n,
    const double eps,
    const size_t min_pts
) {
    bool *h_core = (bool *) malloc_s(n * sizeof(bool));
    if (!h_core) return;

    int *d_cluster;
    int *d_cluster_counter;
    bool *d_core;
    double *d_x, *d_y;
    size_t *d_neighbor_count;
    size_t *d_frontier, *d_next_frontier;
    size_t *d_next_frontier_size;

    CUDA_CHECK(cudaMalloc(&d_cluster, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_core, n * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_neighbor_count, n * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier, n * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, n * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier_size, sizeof(size_t)));

    CUDA_CHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x);

    init_cluster_kernel<<<grid, block>>>(d_cluster, n);
    count_neighbors<<<grid, block>>>(d_neighbor_count, d_core, d_x, d_y, n, eps, min_pts);

    CUDA_CHECK(cudaMemcpy(h_core, d_core, n * sizeof(bool), cudaMemcpyDeviceToHost));

    int cluster_id = NO_CLUSTER;

    for (size_t p = 0; p < n; p++) {
        if (!h_core[p]) continue;

        int h_cluster;
        CUDA_CHECK(cudaMemcpy(&h_cluster, d_cluster + p, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_cluster != NO_CLUSTER) continue;

        cluster_id++;

        // Assign cluster to core point
        CUDA_CHECK(cudaMemcpy(d_cluster + p, &cluster_id, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_frontier, &p, sizeof(size_t), cudaMemcpyHostToDevice));

        size_t frontier_size = 1;
        while (frontier_size > 0) {
            CUDA_CHECK(cudaMemset(d_next_frontier_size, 0, sizeof(size_t)));
            dim3 grid_frontier((frontier_size + block.x - 1) / block.x);

            // Expand frontier
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

            CUDA_CHECK(cudaMemcpy(&frontier_size, d_next_frontier_size, sizeof(size_t), cudaMemcpyDeviceToHost));
            std::swap(d_frontier, d_next_frontier);
        }
    }

    *cluster_count = cluster_id;

    CUDA_CHECK(cudaMemcpy(cluster, d_cluster, n * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_cluster));
    CUDA_CHECK(cudaFree(d_neighbor_count));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier_size));

    cudaDeviceReset();
}
