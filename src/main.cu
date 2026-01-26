#include <cassert>
#include <cstdio>
#include <ctime>
#include "helper.h"
#include "io.h"
#include "common.h"
#include "cpu_dbscan.h"
#include "gpu_dbscan.h"

void free_resource(double **points, int **cluster_cpu, int **cluster_gpu) {
    if (*points) free(*points);
    if (*cluster_cpu) free(*cluster_cpu);
    if (*cluster_gpu) free(*cluster_gpu);
    *points = nullptr;
    *cluster_cpu = nullptr;
    *cluster_gpu = nullptr;
}

void assertion(
    const int* cluster_cpu,
    const int* cluster_gpu,
    const int cluster_count_cpu,
    const int cluster_count_gpu,
    const int n
) {
    assert(cluster_count_cpu == cluster_count_gpu);

    for (int i = 0; i < n; i++) {
        assert(cluster_cpu[i] == cluster_gpu[i]);
    }
}

int test() {
    double *points;
    int n;

    if (!parse_dataset_file(INPUT_FILE, &points, &n)) {
        fprintf(stderr, "Error parsing points file\n");
        return EXIT_FAILURE;
    }

    int *cluster = (int *) malloc_s(n * sizeof(int));
    int cluster_count;
    if (!cluster) {
        free_resource(&points, &cluster, nullptr);
        return EXIT_FAILURE;
    }

    clock_t start, end;
    double elapsed;

    start = clock();
    dbscan_cpu(cluster, &cluster_count, points, n, EPSILON, MIN_PTS);
    end = clock();
    elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    printf("elapsed time: %f seconds\n", elapsed);
    write_dbscan_file(OUTPUT_FILE_CPU, points, cluster, n);

    free_resource(&points, &cluster, nullptr);
    return EXIT_SUCCESS;
}

int hd_run() {
    double *points;
    int n;

    if (!parse_dataset_file(INPUT_FILE, &points, &n)) {
        fprintf(stderr, "Error parsing points file\n");
        return EXIT_FAILURE;
    }

    printf("Processing %s with %u points.\n", INPUT_FILE, n);

    int *cluster_cpu = (int *) malloc_s(n * sizeof(int));
    int *cluster_gpu = (int *) malloc_s(n * sizeof(int));
    int cluster_count_cpu;
    int cluster_count_gpu;
    if (!cluster_cpu || !cluster_gpu) {
        free_resource(&points, &cluster_cpu, &cluster_gpu);
        return EXIT_FAILURE;
    }

    printf("Running sequential DBSCAN...\n");
    clock_t start = clock();
    dbscan_cpu(cluster_cpu, &cluster_count_cpu, points, n, EPSILON, MIN_PTS);
    clock_t end = clock();
    const double elapsed_cpu = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Sequential DBSCAN elapsed time: %f s\n", elapsed_cpu);
    write_dbscan_file(OUTPUT_FILE_CPU, points, cluster_cpu, n);

    printf("Running sequential DBSCAN...\n");
    start = clock();
    dbscan_gpu(cluster_gpu, &cluster_count_gpu, points, n, EPSILON, MIN_PTS);
    end = clock();
    const double elapsed_gpu = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Parallel DBSCAN elapsed time: %f s\n", elapsed_gpu);
    write_dbscan_file(OUTPUT_FILE_GPU, points, cluster_gpu, n);

    const double speedup = elapsed_cpu / elapsed_gpu;
    printf("Speedup: %f\n", speedup);

    assertion(cluster_cpu, cluster_gpu, cluster_count_cpu, cluster_count_gpu, n);

    free_resource(&points, &cluster_cpu, &cluster_gpu);
    return EXIT_SUCCESS;
}

int main() {
    return hd_run();
}
