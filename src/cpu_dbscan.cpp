#include <queue>
#include "cpu_dbscan.h"
#include "common.h"
#include "helper.h"

static void free_dbscan_resources(Grid *grid, int **queue, int **neighbors_buf) {
    free_grid(grid);
    if (*queue) free(*queue);
    if (*neighbors_buf) free(*neighbors_buf);
    *queue = nullptr;
    *neighbors_buf = nullptr;
}

bool init_grid(Grid *grid, const double *x, const double *y, const int n, const double eps) {
    double x_min = x[0], x_max = x[0];
    double y_min = y[0], y_max = y[0];
    for (int i = 1; i < n; i++) {
        x_min = fmin(x_min, x[i]);
        x_max = fmax(x_max, x[i]);
        y_min = fmin(y_min, y[i]);
        y_max = fmax(y_max, y[i]);
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
        int cx, cy;
        cell_coordinates(&cx, &cy, x[i], y[i], x_min, y_min, eps);
        const int c = cy * grid->width + cx;
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
        int cx, cy;
        cell_coordinates(&cx, &cy, x[i], y[i], x_min, y_min, eps);
        const int c = cy * grid->width + cx;
        const int pos = grid->cell_start[c] + grid->cell_offset[c];
        grid->cell_points[pos] = i;
        grid->cell_offset[c]++;
    }

    return true;
}

int find_neighbors(
    int *neighbor,
    const Grid *grid,
    const double *x,
    const double *y,
    const int i
) {
    int count = 0;
    int cx, cy;
    cell_coordinates(&cx, &cy, x[i], y[i], grid->x_min, grid->y_min, grid->cell_size);

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            const int nx = cx + dx;
            const int ny = cy + dy;
            if (nx < 0 || ny < 0 || nx >= grid->width || ny >= grid->height) continue;

            const int c = ny * grid->width + nx;
            const int start = grid->cell_start[c];
            const int offset = grid->cell_offset[c];
            for (int k = 0; k < offset; k++) {
                const int j = grid->cell_points[start + k];
                if (i != j && is_eps_neighbor(x[i], y[i], x[j], y[j], grid->cell_size)) {
                    neighbor[count++] = j;
                }
            }
        }
    }
    return count;
}

void expand_cluster(
    int *cluster,
    int *queue,
    int *neighbors_buf,
    const Grid *grid,
    const double *x,
    const double *y,
    const int min_pts,
    const int cluster_id,
    const int i
) {
    cluster[i] = cluster_id;
    int head = 0, tail = 0;
    queue[tail++] = i;

    while (head < tail) {
        const int j = queue[head++];
        const int degree = find_neighbors(neighbors_buf, grid, x, y, j);

        if (degree < min_pts) continue;

        for (int k = 0; k < degree; k++) {
            const int r = neighbors_buf[k];
            if (cluster[r] == NO_CLUSTER) {
                cluster[r] = cluster_id;
                queue[tail++] = r;
            }
        }
    }
}

void dbscan_cpu(
    int *cluster,
    int *cluster_count,
    const double *x,
    const double *y,
    const int n,
    const double eps,
    const int min_pts
) {
    memset(cluster, NO_CLUSTER, n * sizeof(int));

    Grid grid;
    if (!init_grid(&grid, x, y, n, eps)) {
        fprintf(stderr, "Failed to compute grid\n");
        return;
    }

    int *queue = (int *) malloc_s(n * sizeof(int));
    int *neighbors_buf = (int *) malloc_s(n * sizeof(int));
    if (!queue || !neighbors_buf) {
        free_dbscan_resources(&grid, &queue, &neighbors_buf);
        return;
    }

    int cluster_id = NO_CLUSTER;

    for (int i = 0; i < n; i++) {
        if (cluster[i] != NO_CLUSTER) continue;

        const int degree = find_neighbors(neighbors_buf, &grid, x, y, i);
        if (degree < min_pts) continue;

        expand_cluster(cluster, queue, neighbors_buf, &grid, x, y, min_pts, ++cluster_id, i);
    }

    *cluster_count = cluster_id;

    free_dbscan_resources(&grid, &queue, &neighbors_buf);
}
