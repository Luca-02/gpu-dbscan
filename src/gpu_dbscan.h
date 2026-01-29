#ifndef GPU_DBSCAN_H
#define GPU_DBSCAN_H

/**
 * @brief Performs DBSCAN clustering on a set of 2D points using the GPU.
 * Each point is assigned a cluster label or `NO_CLUSTER_LABEL` if it is considered noise.
 *
 * @param cluster Each element is the cluster label for the corresponding point.
 * @param clusterCount Pointer to an integer where the number of clusters found will be stored.
 * @param x Pointer to the array of x coordinates corresponding to each point.
 * @param y Pointer to the array of y coordinates corresponding to each point.
 * @param n Number of points.
 * @param eps Neighborhood radius for DBSCAN.
 * @param minPts Minimum number of points required to form a core point.
 */
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
