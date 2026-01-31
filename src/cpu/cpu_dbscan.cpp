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
static void cleanup(Grid *grid, int **queue, int **neighbors) {
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
bool initGrid(Grid *grid, const float *x, const float *y, const int n, const float eps) {
    float xMin = x[0], xMax = x[0];
    float yMin = y[0], yMax = y[0];

    for (int i = 1; i < n; i++) {
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

    const int cellCount = grid->width * grid->height;
    const float invEps = 1.0 / grid->eps;

    grid->cellStart = (int *) malloc_s(cellCount * sizeof(int));
    grid->cellSize = (int *) calloc_s(cellCount, sizeof(int));
    grid->cellPoints = (int *) malloc_s(n * sizeof(int));
    if (!grid->cellStart || !grid->cellSize || !grid->cellPoints) {
        freeGrid(grid);
        return false;
    }

    // Compute the cells offsets
    for (int i = 0; i < n; i++) {
        int cx, cy;
        pointCellCoordinates(
            &cx, &cy, x[i], y[i],
            grid->xMin, grid->yMin, invEps
        );

        const int c = linearCellId(cx, cy, grid->width);
        grid->cellSize[c]++;
    }

    // Compute the cells start indexes
    int offset = 0;
    for (int c = 0; c < cellCount; c++) {
        grid->cellStart[c] = offset;
        offset += grid->cellSize[c];
        grid->cellSize[c] = 0;
    }

    // Insert points into its cell
    for (int i = 0; i < n; i++) {
        int cx, cy;
        pointCellCoordinates(
            &cx, &cy, x[i], y[i],
            grid->xMin, grid->yMin, invEps
        );

        const int c = linearCellId(cx, cy, grid->width);
        const int pos = grid->cellStart[c] + grid->cellSize[c];
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
int findNeighbors(
    int *neighbor,
    const float *x,
    const float *y,
    const Grid *grid,
    const int i
) {
    const float invEps = 1.0 / grid-> eps;
    const float eps2 = grid->eps * grid->eps;

    int cx, cy;
    pointCellCoordinates(
        &cx, &cy, x[i], y[i],
        grid->xMin, grid->yMin, invEps
    );

    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= grid->width || ny >= grid->height) continue;

            const int nid = linearCellId(nx, ny, grid->width);
            const int start = grid->cellStart[nid];
            const int end = start + grid->cellSize[nid];

            for (int k = start; k < end; k++) {
                const int j = grid->cellPoints[k];

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
    int *cluster,
    int *queue,
    int *neighbors,
    const float *x,
    const float *y,
    const Grid *grid,
    const int minPts,
    const int clusterId,
    const int i
) {
    cluster[i] = clusterId;
    int head = 0, tail = 0;
    queue[tail++] = i;

    while (head < tail) {
        const int j = queue[head++];
        const int degree = findNeighbors(neighbors, x, y, grid, j);

        if (!isCore(degree, minPts)) continue;

        for (int k = 0; k < degree; k++) {
            const int r = neighbors[k];

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
    int *cluster,
    int *clusterCount,
    const float *x,
    const float *y,
    const int n,
    const float eps,
    const int minPts
) {
    memset(cluster, NO_CLUSTER_LABEL, n * sizeof(int));

    Grid grid;
    if (!initGrid(&grid, x, y, n, eps)) {
        fprintf(stderr, "Failed to compute grid\n");
        return;
    }

    int *queue = (int *) malloc_s(n * sizeof(int));
    int *neighbors = (int *) malloc_s(n * sizeof(int));
    if (!queue || !neighbors) {
        cleanup(&grid, &queue, &neighbors);
        return;
    }

    int clusterId = NO_CLUSTER_LABEL;

    for (int i = 0; i < n; i++) {
        if (cluster[i] != NO_CLUSTER_LABEL) continue;

        const int degree = findNeighbors(neighbors, x, y, &grid, i);
        if (!isCore(degree, minPts)) continue;

        expandCluster(cluster, queue, neighbors, x, y, &grid, minPts, ++clusterId, i);
    }

    *clusterCount = clusterId;

    cleanup(&grid, &queue, &neighbors);
}
