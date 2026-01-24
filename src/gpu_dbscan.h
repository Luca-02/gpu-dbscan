#ifndef GPU_DBSCAN_H
#define GPU_DBSCAN_H

__global__ void binning(
    int *cell_ids,
    int *cell_points,
    const double *x,
    const double *y,
    double x_min,
    double y_min,
    int grid_width,
    int n,
    double eps
);

__global__ void bin_extremes(
    int *cell_starts,
    int *cell_offsets,
    const int *cell_ids,
    int n
);

__global__ void neighbor_counts(
    int *neighbor_counts,
    const double *x,
    const double *y,
    const int *cell_ids,
    const int *cell_starts,
    const int *cell_offsets,
    const int *cell_points,
    int grid_width,
    int grid_height,
    int n,
    double eps
);

__global__ void bfs_expand(
    int *cluster,
    int *next_frontier,
    int *next_frontier_size,
    const double *x,
    const double *y,
    const int *cell_starts,
    const int *cell_offsets,
    const int *cell_points,
    const int *neighbor_counts,
    const int *frontier,
    double x_min,
    double y_min,
    int grid_width,
    int grid_height,
    int frontier_size,
    int cluster_id,
    double eps,
    int min_pts
);

void dbscan_gpu(
    int *cluster,
    int *cluster_count,
    const double *x,
    const double *y,
    int n,
    double eps,
    int min_pts
);

#endif