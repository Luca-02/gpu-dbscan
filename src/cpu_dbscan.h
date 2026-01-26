#ifndef CPU_DBSCAN_H
#define CPU_DBSCAN_H

/**
 * @struct Grid
 * @brief Uniform grid structure used to accelerate neighbor searches in DBSCAN.
 *
 * The grid partitions the 2D space into square cells of size `cell_size`.
 * Each cell stores indices of points that belong to it. This allows
 * efficient lookup of neighbors within a given radius.
 */
typedef struct {
    int *cell_start;    /**< Starting index of points for each cell in `cell_points` array */
    int *cell_offset;   /**< Number of points currently in each cell */
    int *cell_points;   /**< Array storing indices of points for all cells */
    int width;          /**< Number of cells along the x-axis */
    int height;         /**< Number of cells along the y-axis */
    double cell_size;   /**< Size of each cell (corresponds to epsilon) */
    double x_min;       /**< Minimum x coordinate of the points */
    double y_min;       /**< Minimum y coordinate of the points */
} Grid;

/**
 * @brief Frees the memory allocated for the grid.
 *
 * @param grid Pointer to the grid to free.
 */
inline void free_grid(Grid *grid) {
    if (grid->cell_start) free(grid->cell_start);
    if (grid->cell_offset) free(grid->cell_offset);
    if (grid->cell_points) free(grid->cell_points);
    grid->cell_start = nullptr;
    grid->cell_offset = nullptr;
    grid->cell_points = nullptr;
}

/**
 * @brief Performs DBSCAN clustering on a set of 2D points using the CPU.
 * Each point is assigned a cluster label or `NO_CLUSTER_LABEL` if it is considered noise.
 *
 * @param cluster Each element is the cluster label for the corresponding point.
 * @param cluster_count Pointer to an integer where the number of clusters found will be stored.
 * @param points Array of size `2 * n` containing the 2D coordinates of points (x0, y0, x1, y1, ...).
 * @param n Number of points.
 * @param eps Neighborhood radius for DBSCAN.
 * @param min_pts Minimum number of points required to form a core point.
 *
 * @note The array cluster and points must have [n] and [n * 2] elements.
 */
void dbscan_cpu(
    int *cluster,
    int *cluster_count,
    const double *points,
    int n,
    double eps,
    int min_pts
);

#endif