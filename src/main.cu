#include <cassert>
#include <cstdio>
#include <ctime>
#include <string>
#include "helper.h"
#include "./io.h"
#include "./common.h"
#include "./cpu/cpu_dbscan.h"
#include "./gpu/gpu_dbscan.h"

typedef void (*dbscanFn)(uint32_t *, uint32_t *, const float *, const float *, uint32_t, float, uint32_t);

static void freeDatasetNames(char ***datasetNames, const uint32_t fileCount) {
    if (*datasetNames) {
        for (uint32_t i = 0; i < fileCount; i++) {
            if ((*datasetNames)[i]) free((*datasetNames)[i]);
            (*datasetNames)[i] = nullptr;
        }
        free(*datasetNames);
    }
    *datasetNames = nullptr;
}

static void freePoints(float **x, float **y) {
    if (*x) free(*x);
    if (*y) free(*y);
    *x = nullptr;
    *y = nullptr;
}

static void freeClusters(uint32_t **clusterCpu, uint32_t **clusterGpu) {
    if (*clusterCpu) free(*clusterCpu);
    if (*clusterGpu) free(*clusterGpu);
    *clusterCpu = nullptr;
    *clusterGpu = nullptr;
}

static void assertion(
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
    printf("Assertion passed\n");
}

static bool getDatasetNames(char ***datasetNames, uint32_t *fileCount) {
    if (!listFilesInFolder(DATA_IN_PATH, datasetNames, fileCount)) {
        fprintf(stderr, "Error listing dataset names in folder %s\n", DATA_IN_PATH);
        return false;
    }

    qsort(*datasetNames, *fileCount, sizeof(char *), compareDatasetNames);
    return true;
}

static bool loadDataset(const char *datasetName, float **x, float **y, uint32_t *n) {
    if (!parseDatasetFile(datasetName, x, y, n)) {
        fprintf(stderr, "Error parsing points file\n");
        return false;
    }

    printf("Processing %s [n: %d, eps: %f, min_pts:%d].\n",
           datasetName, *n, EPSILON, MIN_PTS);
    return true;
}

static double runDbscan(
    uint32_t *cluster,
    uint32_t *clusterCount,
    const DbscanExecType execType,
    const dbscanFn dbscan,
    const float *x,
    const float *y,
    const uint32_t n,
    const char *inputFile
) {
    const char *label = execType == DbscanExecType::CPU ? "CPU" : "GPU";

    printf("Running %s DBSCAN...\n", label);

    const clock_t start = clock();
    dbscan(cluster, clusterCount, x, y, n, EPSILON, MIN_PTS);
    const clock_t end = clock();

    const double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    printf("%s DBSCAN elapsed time: %f s\n", label, elapsed);

    if (inputFile) {
        if (char *outputFile = makeDbscanOutputName(inputFile, execType)) {
            printf("Writing %s DBSCAN output to %s in %s\n", label, outputFile, DATA_OUT_PATH);
            writeDbscanFile(DATA_OUT_PATH, outputFile, x, y, cluster, n);
            free(outputFile);
        }
    }

    return elapsed;
}

static bool hdDbscanRun(const char *datasetName) {
    float *x, *y;
    uint32_t n;

    const std::string path = std::string(DATA_IN_PATH) + datasetName;

    if (!loadDataset(path.c_str(), &x, &y, &n)) {
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
        DbscanExecType::CPU,
        dbscanCpu,
        x, y, n,
        datasetName
    );

    const double elapsed_gpu = runDbscan(
        cluster_gpu,
        &cluster_count_gpu,
        DbscanExecType::GPU,
        dbscan_gpu,
        x, y, n,
        datasetName
    );

    printf("Speedup: %f\n", elapsed_cpu / elapsed_gpu);

    assertion(cluster_cpu, cluster_gpu, cluster_count_cpu, cluster_count_gpu, n);

    freePoints(&x, &y);
    freeClusters(&cluster_cpu, &cluster_gpu);
    return true;
}

static int testGpu() {
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
        DbscanExecType::GPU,
        dbscan_gpu,
        x, y, n,
        TEST_INPUT_DATASET
    );

    freePoints(&x, &y);
    freeClusters(&cluster_cpu, &cluster_gpu);
    return EXIT_SUCCESS;
}

int main() {
    return testGpu();
    // char **datasetNames;
    // uint32_t fileCount;
    //
    // if (!getDatasetNames(&datasetNames, &fileCount)) {
    //     freeDatasetNames(&datasetNames, fileCount);
    //     return EXIT_FAILURE;
    // }
    //
    // printf("Processing %d datasets\n", fileCount);
    //
    // uint32_t i = 0;
    // while (i < fileCount) {
    //     printf("==================================================\n");
    //     if (!hdDbscanRun(datasetNames[i])) {
    //         break;
    //     }
    //     i++;
    // }
    // printf("==================================================\n");
    //
    // freeDatasetNames(&datasetNames, fileCount);
    //
    // if (i != fileCount) {
    //     return EXIT_FAILURE;
    // }
    //
    // return EXIT_SUCCESS;
}
