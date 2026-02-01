#ifndef CPU_DBSCAN_H
#define CPU_DBSCAN_H

/**
 * @struct Grid
 * @brief Uniform grid structure used to accelerate neighbor searches in DBSCAN.
 *
 * The grid partitions the 2D space into square cells of epsilon size.
 * Each cell stores indices of points that belong to it. This allows
 * efficient lookup of neighbors within a given radius.
 */
typedef struct {
    uint32_t *cellStart;    /**< Starting index of points for each cell in `cellPoints` array */
    uint32_t *cellSize;     /**< Number of points currently in each cell */
    uint32_t *cellPoints;   /**< Array storing indices of points for all cells */

    uint32_t width;         /**< Number of cells along the x-axis */
    uint32_t height;        /**< Number of cells along the y-axis */
    float eps;              /**< Size of each cell */
    float xMin;             /**< Minimum x coordinate of the points */
    float yMin;             /**< Minimum y coordinate of the points */
} Grid;

/**
 * @brief Frees the memory allocated for the grid.
 *
 * @param grid Pointer to the grid to free.
 */
inline void freeGrid(Grid *grid) {
    if (grid->cellStart) free(grid->cellStart);
    if (grid->cellSize) free(grid->cellSize);
    if (grid->cellPoints) free(grid->cellPoints);
    grid->cellStart = nullptr;
    grid->cellSize = nullptr;
    grid->cellPoints = nullptr;
}

/**
 * @brief Performs DBSCAN clustering on a set of 2D points using the CPU.
 * Each point is assigned a cluster label or `NO_CLUSTER_LABEL` if it is considered noise.
 *
 * @param cluster Each element is the cluster label for the corresponding point.
 * @param clusterCount Pointer to an integer where the number of clusters found will be stored.
 * @param x Pointer to the array of x coordinates corresponding to each point.
 * @param y Pointer to the array of y coordinates corresponding to each point.
 * @param n Number of points.
 * @param eps Neighborhood radius for DBSCAN.
 * @param minPts Minimum number of points required to form a core point.
 *
 * @note The array cluster and points must have [n] and [n * 2] elements.
 */
void dbscanCpu(
    uint32_t *cluster,
    uint32_t *clusterCount,
    const float *x,
    const float *y,
    uint32_t n,
    float eps,
    uint32_t minPts
);

#endif