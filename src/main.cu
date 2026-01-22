#include <cassert>
#include <cstdio>
#include <ctime>
#include "helper.h"
#include "io.h"
#include "common.h"
#include "cpu_dbscan.h"
#include "gpu_dbscan.h"

void free_resource(double **x, double **y, int **cluster) {
    if (*x) free(*x);
    if (*y) free(*y);
    if (*cluster) free(*cluster);
    *x = nullptr;
    *y = nullptr;
    *cluster = nullptr;
}

int test() {
    double *x, *y;
    int n;

    if (!parse_input_file(INPUT_FILE, &x, &y, &n)) {
        fprintf(stderr, "Error parsing points file\n");
        return EXIT_FAILURE;
    }

    printf("Processing %u points.\n", n);
    printf("==============================\n");

    int *cluster = (int *) malloc_s(n * sizeof(int));
    int cluster_count;
    if (!cluster) {
        free_resource(&x, &y, &cluster);
        return EXIT_FAILURE;
    }

    clock_t start, end;
    double elapsed;

    start = clock();
    dbscan_gpu(cluster, &cluster_count, x, y, n, EPSILON, MIN_PTS);
    end = clock();
    elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    printf("elapsed time: %f seconds\n", elapsed);
    write_output_file("../data/output_gpu.csv", x, y, cluster, n);

    free_resource(&x, &y, &cluster);
    return EXIT_SUCCESS;
}

int hd_run() {
    double *x, *y;
    int n;

    if (!parse_input_file(INPUT_FILE, &x, &y, &n)) {
        fprintf(stderr, "Error parsing points file\n");
        return EXIT_FAILURE;
    }

    printf("Processing %u points.\n", n);
    printf("==============================\n");

    int *cluster = (int *) malloc_s(n * sizeof(int));
    if (!cluster) {
        free_resource(&x, &y, &cluster);
        return EXIT_FAILURE;
    }

    printf("Running sequential DBSCAN...\n");
    int cluster_count_cpu;
    clock_t start = clock();
    dbscan_cpu(cluster, &cluster_count_cpu, x, y, n, EPSILON, MIN_PTS);
    clock_t end = clock();
    const double elapsed_cpu = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Sequential DBSCAN elapsed time: %f seconds\n", elapsed_cpu);
    write_output_file(OUTPUT_FILE_CPU, x, y, cluster, n);

    printf("==============================\n");

    printf("Running sequential DBSCAN...\n");
    int cluster_count_gpu;
    start = clock();
    dbscan_gpu(cluster, &cluster_count_gpu, x, y, n, EPSILON, MIN_PTS);
    end = clock();
    const double elapsed_gpu = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Parallel DBSCAN elapsed time: %f seconds\n", elapsed_gpu);
    write_output_file(OUTPUT_FILE_GPU, x, y, cluster, n);

    const double speedup = elapsed_cpu / elapsed_gpu;
    printf("Speedup: %f\n", speedup);

    assert(cluster_count_cpu == cluster_count_gpu);

    free_resource(&x, &y, &cluster);
    return EXIT_SUCCESS;
}

int main() {
    const int ret = test();
    return ret;
}
