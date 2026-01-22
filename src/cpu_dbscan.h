#ifndef CPU_DBSCAN_H
#define CPU_DBSCAN_H

typedef struct {
    int width, height;
    double cell_size;
    double x_min, y_min;
    int *cell_start;
    int *cell_offset;
    int *cell_points;
} Grid;

inline void free_grid(Grid *grid) {
    if (grid->cell_start) free(grid->cell_start);
    if (grid->cell_offset) free(grid->cell_offset);
    if (grid->cell_points) free(grid->cell_points);
    grid->cell_start = nullptr;
    grid->cell_offset = nullptr;
    grid->cell_points = nullptr;
}

bool init_grid(
    Grid *grid,
    const double *x,
    const double *y,
    int n,
    double eps
);

int find_neighbors(
    int *neighbor,
    const Grid *grid,
    const double *x,
    const double *y,
    int i
);

void expand_cluster(
    int *cluster,
    int *queue,
    int *neighbors_buf,
    const Grid *grid,
    const double *x,
    const double *y,
    int min_pts,
    int cluster_id,
    int i
);

void dbscan_cpu(
    int *cluster,
    int *cluster_count,
    const double *x,
    const double *y,
    int n,
    double eps,
    int min_pts
);

#endif