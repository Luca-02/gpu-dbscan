#ifndef GPU_DBSCAN_H
#define GPU_DBSCAN_H

void dbscan_gpu(
    int *cluster,
    int *clusterCount,
    const double *x,
    const double *y,
    int n,
    double eps,
    int minPts
);

#endif
