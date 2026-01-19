#include "dbscan.h"


__global__ void kernel_dbscan_gpu(bool *result) {
    result[0] = is_core(10, 5);
}

void dbscan_gpu(int *cluster, const double *x, const double *y, const int n, const double eps, const int min_pts) {
    bool *result;
    cudaMallocManaged(&result, sizeof(bool));
    kernel_dbscan_gpu<<<1, 1>>>(result);
    cudaDeviceSynchronize();
}
