#include <cassert>
#include <cstdio>
#include <ctime>
#include <string>
#include "helper.h"
#include "cuda_helper.h"
#include "io.h"
#include "common.h"
#include "cpu_dbscan.h"
#include "gpu_dbscan.h"

typedef void (*dbscanFn)(int *, int *, const double *, const double *, int, double, int);

void freeResource(double **x, double **y, int **clusterCpu, int **clusterGpu) {
    if (*x) free(*x);
    if (*y) free(*y);
    if (*clusterCpu) free(*clusterCpu);
    if (*clusterGpu) free(*clusterGpu);
    *x = nullptr;
    *y = nullptr;
    *clusterCpu = nullptr;
    *clusterGpu = nullptr;
}

void assertion(
    const int *clusterCpu,
    const int *clusterGpu,
    const int clusterCountCpu,
    const int clusterCountGpu,
    const int n
) {
    assert(clusterCountCpu == clusterCountGpu);

    for (int i = 0; i < n; i++) {
        assert(clusterCpu[i] == clusterGpu[i]);
    }
}

bool loadDataset(double **x, double **y, int *n) {
    if (!parseDatasetFile(INPUT_FILE, x, y, n)) {
        fprintf(stderr, "Error parsing points file\n");
        return false;
    }
    printf("Processing %s [n: %d, eps: %f, min_pts:%d].\n",
        INPUT_FILE, *n, EPSILON, MIN_PTS);
    return true;
}

double runDbscan(
    int *cluster,
    int *clusterCount,
    const char *label,
    const dbscanFn dbscan,
    const double *x,
    const double *y,
    const int n,
    const char *outputFile
) {
    printf("Running %s DBSCAN...\n", label);

    const clock_t start = clock();
    dbscan(cluster, clusterCount, x, y, n, EPSILON, MIN_PTS);
    const clock_t end = clock();

    const double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    printf("%s DBSCAN elapsed time: %f s\n", label, elapsed);

    writeDbscanFile(outputFile, x, y, cluster, n);
    return elapsed;
}

int test_gpu() {
    double *x, *y;
    int n;

    if (!loadDataset(&x, &y, &n)) {
        return EXIT_FAILURE;
    }

    int *cluster_cpu = (int *) malloc_s(n * sizeof(int));
    int *cluster_gpu = (int *) malloc_s(n * sizeof(int));
    if (!cluster_cpu || !cluster_gpu) {
        freeResource(&x, &y, &cluster_cpu, &cluster_gpu);
        return EXIT_FAILURE;
    }

    int cluster_count_gpu;

    runDbscan(
        cluster_gpu,
        &cluster_count_gpu,
        "Parallel",
        dbscan_gpu,
        x, y, n,
        OUTPUT_FILE_GPU
    );

    freeResource(&x, &y, &cluster_cpu, &cluster_gpu);
    return EXIT_SUCCESS;
}

int test_read_files() {
    char **fileNames;
    int fileCount;

    if (!listFilesInFolder(DATA_IN_PATH, &fileNames, &fileCount)) {
        for (int i = 0; i < fileCount; i++) {
            if (fileNames[i]) free(fileNames[i]);
            fileNames[i] = nullptr;
        }
        return EXIT_FAILURE;
    }

    qsort(fileNames, fileCount, sizeof(char *), compareStrings);

    for (int i = 0; i < fileCount; i++) {
        printf("%s\n", fileNames[i]);
    }

    for (int i = 0; i < fileCount; i++) {
        if (fileNames[i]) free(fileNames[i]);
        fileNames[i] = nullptr;
    }
    return EXIT_SUCCESS;
}

int hd_run() {
    double *x, *y;
    int n;

    if (!loadDataset(&x, &y, &n)) {
        freeResource(&x, &y, nullptr, nullptr);
        return EXIT_FAILURE;
    }

    int *cluster_cpu = (int *) malloc_s(n * sizeof(int));
    int *cluster_gpu = (int *) malloc_s(n * sizeof(int));
    if (!cluster_cpu || !cluster_gpu) {
        freeResource(&x, &y, &cluster_cpu, &cluster_gpu);
        return EXIT_FAILURE;
    }

    int cluster_count_cpu, cluster_count_gpu;

    const double elapsed_cpu = runDbscan(
        cluster_cpu,
        &cluster_count_cpu,
        "Sequential",
        dbscanCpu,
        x, y, n,
        OUTPUT_FILE_CPU
    );

    const double elapsed_gpu = runDbscan(
        cluster_gpu,
        &cluster_count_gpu,
        "Parallel",
        dbscan_gpu,
        x, y, n,
        OUTPUT_FILE_GPU
    );

    printf("Speedup: %f\n", elapsed_cpu / elapsed_gpu);

    assertion(cluster_cpu, cluster_gpu, cluster_count_cpu, cluster_count_gpu, n);

    freeResource(&x, &y, &cluster_cpu, &cluster_gpu);
    return EXIT_SUCCESS;
}

// TODO rename all using CamelCase naming convention, better
int main() {
    // deviceFeat();
    return test_read_files();
}
