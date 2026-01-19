#include "dbscan.h"


__global__ void count_neighbors(
    int *neighbor_count,
    const double *x,
    const double *y,
    const int n,
    const double eps2
) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const double xi = x[i];
    const double yi = y[i];

    size_t count = 0;
    for (size_t j = 0; j < n; j++) {
        const double dx = xi - x[j];
        const double dy = yi - y[j];
        if (dx * dx + dy * dy <= eps2) {
            count++;
        }
    }

    neighbor_count[i] = count;
}

void dbscan_gpu(
    int *cluster,
    const double *x,
    const double *y,
    const size_t n,
    const double eps,
    const size_t min_pts
) {

}
