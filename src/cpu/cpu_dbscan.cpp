#include <queue>
#include "./cpu_dbscan.h"
#include "../common.h"
#include "../helper.h"

/**
 * @brief Cleans up resources used for DBSCAN.
 *
 * @param grid Pointer to the Grid structure.
 * @param queue Pointer to the pointer of the queue array.
 * @param neighbors Pointer to the pointer of the neighbors buffer array.
 */
static void cleanup(Grid *grid, uint32_t **queue, uint32_t **neighbors) {
    freeGrid(grid);
    if (*queue) free(*queue);
    if (*neighbors) free(*neighbors);
    *queue = nullptr;
    *neighbors = nullptr;
}

/**
 * @brief Initializes a uniform grid for fast neighbor searches.
 * Computes the bounding box of the points, allocates memory for the
 * grid structure, and assigns each point to its corresponding cell.
 *
 * @param grid Pointer to the Grid to initialize.
 * @param x Pointer to the array of x coordinates corresponding to each point.
 * @param y Pointer to the array of y coordinates corresponding to each point.
 * @param n Number of points.
 * @param eps Neighborhood radius (cell size).
 * @return True if the grid was successfully initialized, false otherwise.
 */
bool initGrid(Grid *grid, const float *x, const float *y, const uint32_t n, const float eps) {
    float xMin = x[0], xMax = x[0];
    float yMin = y[0], yMax = y[0];

    for (uint32_t i = 1; i < n; i++) {
        xMin = fmin(xMin, x[i]);
        xMax = fmax(xMax, x[i]);
        yMin = fmin(yMin, y[i]);
        yMax = fmax(yMax, y[i]);
    }

    grid->xMin = xMin;
    grid->yMin = yMin;
    grid->eps = eps;

    grid->width = ceil((xMax - grid->xMin) / eps + 1);
    grid->height = ceil((yMax - grid->yMin) / eps + 1);

    const uint32_t cellCount = grid->width * grid->height;
    const float invEps = 1.0 / grid->eps;

    grid->cellStart = (uint32_t *) malloc_s(cellCount * sizeof(uint32_t));
    grid->cellSize = (uint32_t *) calloc_s(cellCount, sizeof(uint32_t));
    grid->cellPoints = (uint32_t *) malloc_s(n * sizeof(uint32_t));
    if (!grid->cellStart || !grid->cellSize || !grid->cellPoints) {
        freeGrid(grid);
        return false;
    }

    // Compute the cells offsets
    for (uint32_t i = 0; i < n; i++) {
        uint32_t cx, cy;
        pointCellCoordinates(
            &cx, &cy, x[i], y[i],
            grid->xMin, grid->yMin, invEps
        );

        const uint32_t c = linearCellId(cx, cy, grid->width);
        grid->cellSize[c]++;
    }

    // Compute the cells start indexes
    uint32_t offset = 0;
    for (uint32_t c = 0; c < cellCount; c++) {
        grid->cellStart[c] = offset;
        offset += grid->cellSize[c];
        grid->cellSize[c] = 0;
    }

    // Insert points into its cell
    for (uint32_t i = 0; i < n; i++) {
        uint32_t cx, cy;
        pointCellCoordinates(
            &cx, &cy, x[i], y[i],
            grid->xMin, grid->yMin, invEps
        );

        const uint32_t c = linearCellId(cx, cy, grid->width);
        const uint32_t pos = grid->cellStart[c] + grid->cellSize[c];
        grid->cellPoints[pos] = i;
        grid->cellSize[c]++;
    }

    return true;
}

/**
 * @brief Finds all neighbors of a given point within epsilon distance.
 *
 * @param neighbor Output array to store indices of neighboring points.
 * @param x Pointer to the array of x coordinates corresponding to each point.
 * @param y Pointer to the array of y coordinates corresponding to each point.
 * @param grid Pointer to the precomputed grid.
 * @param i Index of the point for which neighbors are searched.
 * @return Number of neighbors found.
 *
 * @note The array neighbor must have [n] elements.
 */
uint32_t findNeighbors(
    uint32_t *neighbor,
    const float *x,
    const float *y,
    const Grid *grid,
    const uint32_t i
) {
    const float invEps = 1.0 / grid->eps;
    const float eps2 = grid->eps * grid->eps;

    uint32_t cx, cy;
    pointCellCoordinates(
        &cx, &cy, x[i], y[i],
        grid->xMin, grid->yMin, invEps
    );

    uint32_t count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if ((dx < 0 && cx < (uint32_t) -dx) || (dy < 0 && cy < (uint32_t) -dy)) continue;

            const uint32_t nx = cx + dx;
            const uint32_t ny = cy + dy;

            if (nx >= grid->width || ny >= grid->height) continue;

            const uint32_t nid = linearCellId(nx, ny, grid->width);
            const uint32_t start = grid->cellStart[nid];
            const uint32_t end = start + grid->cellSize[nid];

            for (uint32_t k = start; k < end; k++) {
                const uint32_t j = grid->cellPoints[k];

                if (i != j && isEpsNeighbor(x[i], y[i], x[j], y[j], eps2)) {
                    neighbor[count++] = j;
                }
            }
        }
    }

    return count;
}

/**
 * @brief Expands a cluster starting from a core point.
 * The function assigns the cluster label to all reachable points
 * starting from point `i` using a breadth-first expansion.
 *
 * @param cluster Array storing cluster labels for each point.
 * @param queue Temporary queue used for breadth-first search.
 * @param neighbors Temporary buffer for neighbor indices.
 * @param x Pointer to the array of x coordinates corresponding to each point.
 * @param y Pointer to the array of y coordinates corresponding to each point.
 * @param grid Pointer to the precomputed grid.
 * @param minPts Minimum points to form a core point.
 * @param clusterId ID of the cluster being expanded.
 * @param i Index of the core point to start the expansion.
 *
 * @note The arrays neighbor and queue must have [n] elements.
 */
void expandCluster(
    uint32_t *cluster,
    uint32_t *queue,
    uint32_t *neighbors,
    const float *x,
    const float *y,
    const Grid *grid,
    const uint32_t minPts,
    const uint32_t clusterId,
    const uint32_t i
) {
    cluster[i] = clusterId;
    uint32_t head = 0, tail = 0;
    queue[tail++] = i;

    while (head < tail) {
        const uint32_t j = queue[head++];
        const uint32_t degree = findNeighbors(neighbors, x, y, grid, j);

        if (!isCore(degree, minPts)) continue;

        for (uint32_t k = 0; k < degree; k++) {
            const uint32_t r = neighbors[k];

            if (cluster[r] == NO_CLUSTER_LABEL) {
                cluster[r] = clusterId;
                queue[tail++] = r;
            }
        }
    }
}

/**
 * @brief CPU implementation of DBSCAN clustering.
 * This function initializes the grid, allocates buffers, and iterates over all points
 * to form clusters. Core points expand clusters using a breadth-first approach.
 *
 * @param cluster Array to store cluster labels for each point.
 * @param clusterCount Pointer to store the number of clusters found.
 * @param x Pointer to the array of x coordinates corresponding to each point.
 * @param y Pointer to the array of y coordinates corresponding to each point.
 * @param n Number of points.
 * @param eps Neighborhood radius.
 * @param minPts Minimum points to form a core point.
 */
void dbscanCpu(
    uint32_t *cluster,
    uint32_t *clusterCount,
    const float *x,
    const float *y,
    const uint32_t n,
    const float eps,
    const uint32_t minPts
) {
    memset(cluster, NO_CLUSTER_LABEL, n * sizeof(int));

    Grid grid;
    if (!initGrid(&grid, x, y, n, eps)) {
        fprintf(stderr, "Failed to compute grid\n");
        return;
    }

    uint32_t *queue = (uint32_t *) malloc_s(n * sizeof(uint32_t));
    uint32_t *neighbors = (uint32_t *) malloc_s(n * sizeof(uint32_t));
    if (!queue || !neighbors) {
        cleanup(&grid, &queue, &neighbors);
        return;
    }

    uint32_t clusterId = NO_CLUSTER_LABEL;

    for (uint32_t i = 0; i < n; i++) {
        if (cluster[i] != NO_CLUSTER_LABEL) continue;

        const uint32_t degree = findNeighbors(neighbors, x, y, &grid, i);
        if (!isCore(degree, minPts)) continue;

        expandCluster(cluster, queue, neighbors, x, y, &grid, minPts, ++clusterId, i);
    }

    *clusterCount = clusterId;

    cleanup(&grid, &queue, &neighbors);
}
