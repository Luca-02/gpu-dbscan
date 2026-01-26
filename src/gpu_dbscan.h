#ifndef GPU_DBSCAN_H
#define GPU_DBSCAN_H

void dbscan_gpu(
    int *cluster,
    int *cluster_count,
    const double *points,
    int n,
    double eps,
    int min_pts
);

#endif
