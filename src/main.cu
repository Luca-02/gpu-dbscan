#include <cassert>
#include <cstdio>
#include <ctime>
#include <string>
#include "helper.h"
// #include "cuda_helper.h"
#include "io.h"
#include "common.h"
#include "cpu/cpu_dbscan.h"
#include "gpu/gpu_dbscan.h"

typedef void (*dbscanFn)(int *, int *, const float *, const float *, int, float, int);

void freeDatasetNames(char ***datasetNames, const int fileCount) {
    if (*datasetNames) {
        for (int i = 0; i < fileCount; i++) {
            if ((*datasetNames)[i]) free((*datasetNames)[i]);
            (*datasetNames)[i] = nullptr;
        }
        free(*datasetNames);
    }
    *datasetNames = nullptr;
}

void freePoints(float **x, float **y) {
    if (*x) free(*x);
    if (*y) free(*y);
    *x = nullptr;
    *y = nullptr;
}

void freeClusters(int **clusterCpu, int **clusterGpu) {
    if (*clusterCpu) free(*clusterCpu);
    if (*clusterGpu) free(*clusterGpu);
    *clusterCpu = nullptr;
    *clusterGpu = nullptr;
}

void cleanup(
    char ***datasetNames,
    float **x,
    float **y,
    int **clusterCpu,
    int **clusterGpu,
    const int fileCount
) {
    freeDatasetNames(datasetNames, fileCount);
    freePoints(x, y);
    freeClusters(clusterCpu, clusterGpu);
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

bool getDatasetNames(char*** datasetNames, int *fileCount) {
    if (!listFilesInFolder(DATA_IN_PATH, datasetNames, fileCount)) {
        fprintf(stderr, "Error listing dataset names in folder %s\n", DATA_IN_PATH);
        return false;
    }
    qsort(*datasetNames, *fileCount, sizeof(char *), compareDatasetNames);
    return true;
}

bool loadDataset(float **x, float **y, int *n) {
    if (!parseDatasetFile(TEST_INPUT_DATASET, x, y, n)) {
        fprintf(stderr, "Error parsing points file\n");
        return false;
    }
    printf("Processing %s [n: %d, eps: %f, min_pts:%d].\n",
        TEST_INPUT_DATASET, *n, EPSILON, MIN_PTS);
    return true;
}

double runDbscan(
    int *cluster,
    int *clusterCount,
    const char *label,
    const dbscanFn dbscan,
    const float *x,
    const float *y,
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

int testDatasetNames() {
    char **datasetNames;
    int fileCount;

    if (!getDatasetNames(&datasetNames, &fileCount)) {
        freeDatasetNames(&datasetNames, fileCount);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < fileCount; i++) {
        printf("%s\n", datasetNames[i]);
    }

    freeDatasetNames(&datasetNames, fileCount);
    return EXIT_SUCCESS;
}

int testGpu() {
    float *x, *y;
    int n;

    if (!loadDataset(&x, &y, &n)) {
        freePoints(&x, &y);
        return EXIT_FAILURE;
    }

    int *cluster_cpu = (int *) malloc_s(n * sizeof(int));
    int *cluster_gpu = (int *) malloc_s(n * sizeof(int));
    if (!cluster_cpu || !cluster_gpu) {
        freePoints(&x, &y);
        freeClusters(&cluster_cpu, &cluster_gpu);
        return EXIT_FAILURE;
    }

    int cluster_count_gpu;

    runDbscan(
        cluster_gpu,
        &cluster_count_gpu,
        "Parallel",
        dbscan_gpu,
        x, y, n,
        TEST_OUTPUT_DBSCAN_GPU
    );

    freePoints(&x, &y);
    freeClusters(&cluster_cpu, &cluster_gpu);
    return EXIT_SUCCESS;
}

int hdRun() {
    float *x, *y;
    int n;

    if (!loadDataset(&x, &y, &n)) {
        freePoints(&x, &y);
        return EXIT_FAILURE;
    }

    int cluster_count_cpu, cluster_count_gpu;
    int *cluster_cpu = (int *) malloc_s(n * sizeof(int));
    int *cluster_gpu = (int *) malloc_s(n * sizeof(int));
    if (!cluster_cpu || !cluster_gpu) {
        freePoints(&x, &y);
        freeClusters(&cluster_cpu, &cluster_gpu);
        return EXIT_FAILURE;
    }

    const double elapsed_cpu = runDbscan(
        cluster_cpu,
        &cluster_count_cpu,
        "Sequential",
        dbscanCpu,
        x, y, n,
        TEST_OUTPUT_DBSCAN_CPU
    );

    const double elapsed_gpu = runDbscan(
        cluster_gpu,
        &cluster_count_gpu,
        "Parallel",
        dbscan_gpu,
        x, y, n,
        TEST_OUTPUT_DBSCAN_GPU
    );

    printf("Speedup: %f\n", elapsed_cpu / elapsed_gpu);

    assertion(cluster_cpu, cluster_gpu, cluster_count_cpu, cluster_count_gpu, n);

    freePoints(&x, &y);
    freeClusters(&cluster_cpu, &cluster_gpu);
    return EXIT_SUCCESS;
}

int datasetsRun() {
    return EXIT_SUCCESS;
}

// TODO rename all using CamelCase naming convention, better
int main() {
    // deviceFeat();
    return testGpu();
}
