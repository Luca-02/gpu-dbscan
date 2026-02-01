#include <cassert>
#include <cstdio>
#include <ctime>
#include <string>
#include "helper.h"
#include "io.h"
#include "common.h"
#include "cpu/cpu_dbscan.h"
#include "gpu/gpu_dbscan.h"

typedef void (*dbscanFn)(uint32_t *, uint32_t *, const float *, const float *, uint32_t, float, uint32_t);

void freeDatasetNames(char ***datasetNames, const uint32_t fileCount) {
    if (*datasetNames) {
        for (uint32_t i = 0; i < fileCount; i++) {
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

void freeClusters(uint32_t **clusterCpu, uint32_t **clusterGpu) {
    if (*clusterCpu) free(*clusterCpu);
    if (*clusterGpu) free(*clusterGpu);
    *clusterCpu = nullptr;
    *clusterGpu = nullptr;
}

void cleanup(
    char ***datasetNames,
    float **x,
    float **y,
    uint32_t **clusterCpu,
    uint32_t **clusterGpu,
    const uint32_t fileCount
) {
    freeDatasetNames(datasetNames, fileCount);
    freePoints(x, y);
    freeClusters(clusterCpu, clusterGpu);
}

void assertion(
    const uint32_t *clusterCpu,
    const uint32_t *clusterGpu,
    const uint32_t clusterCountCpu,
    const uint32_t clusterCountGpu,
    const uint32_t n
) {
    assert(clusterCountCpu == clusterCountGpu);

    for (uint32_t i = 0; i < n; i++) {
        assert(clusterCpu[i] == clusterGpu[i]);
    }
}

bool getDatasetNames(char ***datasetNames, uint32_t *fileCount) {
    if (!listFilesInFolder(DATA_IN_PATH, datasetNames, fileCount)) {
        fprintf(stderr, "Error listing dataset names in folder %s\n", DATA_IN_PATH);
        return false;
    }

    qsort(*datasetNames, *fileCount, sizeof(char *), compareDatasetNames);
    return true;
}

bool loadDataset(const char *datasetName, float **x, float **y, uint32_t *n) {
    if (!parseDatasetFile(datasetName, x, y, n)) {
        fprintf(stderr, "Error parsing points file\n");
        return false;
    }

    printf("Processing %s [n: %d, eps: %f, min_pts:%d].\n",
           datasetName, *n, EPSILON, MIN_PTS);
    return true;
}

double runDbscan(
    uint32_t *cluster,
    uint32_t *clusterCount,
    const char *label,
    const dbscanFn dbscan,
    const float *x,
    const float *y,
    const uint32_t n,
    const char *outputFile
) {
    printf("Running %s DBSCAN...\n", label);

    const clock_t start = clock();
    dbscan(cluster, clusterCount, x, y, n, EPSILON, MIN_PTS);
    const clock_t end = clock();

    const double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    printf("%s DBSCAN elapsed time: %f s\n", label, elapsed);

    if (outputFile) {
        writeDbscanFile(outputFile, x, y, cluster, n);
    }

    return elapsed;
}

double runDbscan(
    uint32_t *cluster,
    uint32_t *clusterCount,
    const char *label,
    const dbscanFn dbscan,
    const float *x,
    const float *y,
    const uint32_t n
) {
    return runDbscan(cluster, clusterCount, label, dbscan, x, y, n, nullptr);
}

bool hdDbscanRun(const char *datasetName) {
    float *x, *y;
    uint32_t n;

    if (!loadDataset(datasetName, &x, &y, &n)) {
        freePoints(&x, &y);
        return false;
    }

    uint32_t cluster_count_cpu, cluster_count_gpu;
    uint32_t *cluster_cpu = (uint32_t *) malloc_s(n * sizeof(uint32_t));
    uint32_t *cluster_gpu = (uint32_t *) malloc_s(n * sizeof(uint32_t));
    if (!cluster_cpu || !cluster_gpu) {
        freePoints(&x, &y);
        freeClusters(&cluster_cpu, &cluster_gpu);
        return false;
    }

    const double elapsed_cpu = runDbscan(
        cluster_cpu,
        &cluster_count_cpu,
        "Sequential",
        dbscanCpu,
        x, y, n
    );

    const double elapsed_gpu = runDbscan(
        cluster_gpu,
        &cluster_count_gpu,
        "Parallel",
        dbscan_gpu,
        x, y, n
    );

    printf("Speedup: %f\n", elapsed_cpu / elapsed_gpu);

    assertion(cluster_cpu, cluster_gpu, cluster_count_cpu, cluster_count_gpu, n);

    freePoints(&x, &y);
    freeClusters(&cluster_cpu, &cluster_gpu);
    return true;
}

int testGpu() {
    float *x, *y;
    uint32_t n;

    if (!loadDataset(TEST_INPUT_DATASET, &x, &y, &n)) {
        freePoints(&x, &y);
        return EXIT_FAILURE;
    }

    uint32_t *cluster_cpu = (uint32_t *) malloc_s(n * sizeof(uint32_t));
    uint32_t *cluster_gpu = (uint32_t *) malloc_s(n * sizeof(uint32_t));
    if (!cluster_cpu || !cluster_gpu) {
        freePoints(&x, &y);
        freeClusters(&cluster_cpu, &cluster_gpu);
        return EXIT_FAILURE;
    }

    uint32_t cluster_count_gpu;

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

int main() {
    char **datasetNames;
    uint32_t fileCount;

    if (!getDatasetNames(&datasetNames, &fileCount)) {
        freeDatasetNames(&datasetNames, fileCount);
        return EXIT_FAILURE;
    }

    uint32_t i = 0;
    while (i < fileCount) {
        std::string path = std::string(DATA_IN_PATH) + datasetNames[i];
        printf("==================================================\n");
        if (!hdDbscanRun(path.c_str())) {
            break;
        }
        i++;
    }
    printf("==================================================\n");

    freeDatasetNames(&datasetNames, fileCount);

    if (i != fileCount) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
