#include <queue>
#include "cpu_dbscan.h"
#include "common.h"
#include "helper.h"

/**
 * @brief Cleans up resources used for DBSCAN.
 *
 * @param grid Pointer to the Grid structure.
 * @param queue Pointer to the pointer of the queue array.
 * @param neighbors Pointer to the pointer of the neighbors buffer array.
 */
static void cleanup(Grid *grid, int **queue, int **neighbors) {
    free_grid(grid);
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
 * @param points Flattened array of 2D points.
 * @param n Number of points.
 * @param eps Neighborhood radius (cell size).
 * @return True if the grid was successfully initialized, false otherwise.
 */
bool init_grid(Grid *grid, const double *points, const int n, const double eps) {
    double x_min = points[X_INDEX(0)], x_max = points[X_INDEX(0)];
    double y_min = points[Y_INDEX(0)], y_max = points[Y_INDEX(0)];

    for (int i = 1; i < n; i++) {
        x_min = fmin(x_min, points[X_INDEX(i)]);
        x_max = fmax(x_max, points[X_INDEX(i)]);
        y_min = fmin(y_min, points[Y_INDEX(i)]);
        y_max = fmax(y_max, points[Y_INDEX(i)]);
    }

    grid->x_min = x_min;
    grid->y_min = y_min;
    grid->cell_size = eps;

    grid->width = ceil((x_max - grid->x_min) / eps + 1);
    grid->height = ceil((y_max - grid->y_min) / eps + 1);

    const int cell_count = grid->width * grid->height;

    grid->cell_start = (int *) malloc_s(cell_count * sizeof(int));
    grid->cell_offset = (int *) calloc_s(cell_count, sizeof(int));
    grid->cell_points = (int *) malloc_s(n * sizeof(int));
    if (!grid->cell_start || !grid->cell_offset || !grid->cell_points) {
        free_grid(grid);
        return false;
    }

    // Compute the cells offsets
    for (int i = 0; i < n; i++) {
        const double x = points[X_INDEX(i)];
        const double y = points[Y_INDEX(i)];

        int cx, cy;
        point_cell_coordinates(
            &cx, &cy, x, y,
            grid->x_min, grid->y_min, grid->cell_size
        );

        const int c = cell_id(cx, cy, grid->width);
        grid->cell_offset[c]++;
    }

    // Compute the cells start indexes
    int offset = 0;
    for (int c = 0; c < cell_count; c++) {
        grid->cell_start[c] = offset;
        offset += grid->cell_offset[c];
        grid->cell_offset[c] = 0;
    }

    // Insert points into its cell
    for (int i = 0; i < n; i++) {
        const double x = points[X_INDEX(i)];
        const double y = points[Y_INDEX(i)];

        int cx, cy;
        point_cell_coordinates(
            &cx, &cy, x, y,
            grid->x_min, grid->y_min, grid->cell_size
        );

        const int c = cell_id(cx, cy, grid->width);
        const int pos = grid->cell_start[c] + grid->cell_offset[c];
        grid->cell_points[pos] = i;
        grid->cell_offset[c]++;
    }

    return true;
}

/**
 * @brief Finds all neighbors of a given point within epsilon distance.
 *
 * @param neighbor Output array to store indices of neighboring points.
 * @param points Flattened array of 2D points.
 * @param grid Pointer to the precomputed grid.
 * @param i Index of the point for which neighbors are searched.
 * @return Number of neighbors found.
 *
 * @note The array neighbor must have [n] elements.
 */
int find_neighbors(
    int *neighbor,
    const double *points,
    const Grid *grid,
    const int i
) {
    const double xi = points[X_INDEX(i)];
    const double yi = points[Y_INDEX(i)];
    const double eps2 = grid->cell_size * grid->cell_size;

    int cx, cy;
    point_cell_coordinates(
        &cx, &cy, xi, yi,
        grid->x_min, grid->y_min, grid->cell_size
    );

    int count = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= grid->width || ny >= grid->height) continue;

            const int c = cell_id(nx, ny, grid->width);
            const int start = grid->cell_start[c];
            const int offset = grid->cell_offset[c];

            for (int k = 0; k < offset; k++) {
                const int j = grid->cell_points[start + k];
                const double xj = points[X_INDEX(j)];
                const double yj = points[Y_INDEX(j)];

                if (i != j && is_eps_neighbor(xi, yi, xj, yj, eps2)) {
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
 * @param neighbors_buf Temporary buffer for neighbor indices.
 * @param points Flattened array of 2D points.
 * @param grid Pointer to the precomputed grid.
 * @param min_pts Minimum points to form a core point.
 * @param cluster_id ID of the cluster being expanded.
 * @param i Index of the core point to start the expansion.
 *
 * @note The arrays neighbor and queue must have [n] elements.
 */
void expand_cluster(
    int *cluster,
    int *queue,
    int *neighbors_buf,
    const double *points,
    const Grid *grid,
    const int min_pts,
    const int cluster_id,
    const int i
) {
    cluster[i] = cluster_id;
    int head = 0, tail = 0;
    queue[tail++] = i;

    while (head < tail) {
        const int j = queue[head++];
        const int degree = find_neighbors(neighbors_buf, points, grid, j);

        if (!is_core(degree, min_pts)) continue;

        for (int k = 0; k < degree; k++) {
            const int r = neighbors_buf[k];

            if (cluster[r] == NO_CLUSTER_LABEL) {
                cluster[r] = cluster_id;
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
 * @param cluster_count Pointer to store the number of clusters found.
 * @param points Flattened array of 2D points.
 * @param n Number of points.
 * @param eps Neighborhood radius.
 * @param min_pts Minimum points to form a core point.
 */
void dbscan_cpu(
    int *cluster,
    int *cluster_count,
    const double *points,
    const int n,
    const double eps,
    const int min_pts
) {
    memset(cluster, NO_CLUSTER_LABEL, n * sizeof(int));

    Grid grid;
    if (!init_grid(&grid, points, n, eps)) {
        fprintf(stderr, "Failed to compute grid\n");
        return;
    }

    int *queue = (int *) malloc_s(n * sizeof(int));
    int *neighbors_buf = (int *) malloc_s(n * sizeof(int));
    if (!queue || !neighbors_buf) {
        cleanup(&grid, &queue, &neighbors_buf);
        return;
    }

    int cluster_id = NO_CLUSTER_LABEL;

    for (int i = 0; i < n; i++) {
        if (cluster[i] != NO_CLUSTER_LABEL) continue;

        const int degree = find_neighbors(neighbors_buf, points, &grid, i);
        if (!is_core(degree, min_pts)) continue;

        expand_cluster(cluster, queue, neighbors_buf, points, &grid, min_pts, ++cluster_id, i);
    }

    *cluster_count = cluster_id;

    cleanup(&grid, &queue, &neighbors_buf);
}
