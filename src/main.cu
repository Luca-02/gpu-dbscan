#include <cassert>
#include <cstdio>
#include <ctime>
#include "helper.h"
#include "io.h"
#include "dbscan.h"

void free_resource(double **x, double **y, int **cluster) {
    if (*x) {
        free(*x);
        *x = nullptr;
    }
    if (*y) {
        free(*y);
        *y = nullptr;
    }
    if (*cluster) {
        free(*cluster);
        *cluster = nullptr;
    }
}

int main() {
    double *x, *y;
    size_t n;

    if (!parse_input_file(INPUT_FILE, &x, &y, &n)) {
        fprintf(stderr, "Error parsing points file\n");
        return EXIT_FAILURE;
    }

    printf("Processing %zu points.\n", n);
    printf("==============================\n");

    int *cluster = (int *) malloc_s(n * sizeof(int));
    size_t cluster_count_cpu;
    size_t cluster_count_gpu;
    if (!cluster) {
        free_resource(&x, &y, &cluster);
        return EXIT_FAILURE;
    }

    clock_t start = clock();
    dbscan_gpu(cluster, &cluster_count_cpu, x, y, n, EPSILON, MIN_PTS);
    clock_t end = clock();
    const double elapsed_cpu = (double) (end - start) / CLOCKS_PER_SEC;
    printf("dbscan_gpu elapsed time: %f seconds\n", elapsed_cpu);
    write_output_file(OUTPUT_FILE_GPU, x, y, cluster, n);

    free_resource(&x, &y, &cluster);
    return EXIT_SUCCESS;
}


// int main() {
//     double *x, *y;
//     size_t n;
//
//     if (!parse_input_file(INPUT_FILE, &x, &y, &n)) {
//         fprintf(stderr, "Error parsing points file\n");
//         return EXIT_FAILURE;
//     }
//
//     printf("Processing %zu points.\n", n);
//     printf("==============================\n");
//
//     int *cluster = (int *) malloc_s(n * sizeof(int));
//     size_t cluster_count_cpu;
//     size_t cluster_count_gpu;
//     if (!cluster) {
//         free_resource(&x, &y, &cluster);
//         return EXIT_FAILURE;
//     }
//
//     printf("Running sequential DBSCAN...\n");
//     clock_t start = clock();
//     dbscan_cpu(cluster, &cluster_count_cpu, x, y, n, EPSILON, MIN_PTS);
//     clock_t end = clock();
//     const double elapsed_cpu = (double) (end - start) / CLOCKS_PER_SEC;
//     printf("Sequential DBSCAN elapsed time: %f seconds\n", elapsed_cpu);
//     write_output_file(OUTPUT_FILE_CPU, x, y, cluster, n);
//
//     printf("==============================\n");
//
//     printf("Running sequential DBSCAN...\n");
//     start = clock();
//     dbscan_gpu(cluster, &cluster_count_gpu, x, y, n, EPSILON, MIN_PTS);
//     end = clock();
//     const double elapsed_gpu = (double) (end - start) / CLOCKS_PER_SEC;
//     printf("Parallel DBSCAN elapsed time: %f seconds\n", elapsed_gpu);
//     write_output_file(OUTPUT_FILE_GPU, x, y, cluster, n);
//
//     const double speedup = elapsed_cpu / elapsed_gpu;
//     printf("Speedup: %f\n", speedup);
//
//     assert(cluster_count_cpu == cluster_count_gpu);
//
//     free_resource(&x, &y, &cluster);
//     return EXIT_SUCCESS;
// }
