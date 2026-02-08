#ifndef IO_H
#define IO_H
#include "./common.h"

#define MAX_FILES 1024

/**
 * @brief Compares two dataset file names by their dataset number.
 *
 * @param a Pointer to first file name.
 * @param b Pointer to second file name.
 * @return Negative if a < b, positive if a > b, zero if equal.
 */
int compareDatasetNames(const void *a, const void *b);

/**
 * @brief Creates the output file name for the DBSCAN algorithm.
 *
 * @param datasetName The name of the input dataset file.
 * @param execType The execution type of the DBSCAN algorithm.
 * @return The output file name, or nullptr if an error occurs.
 */
char *makeDbscanOutputName(
    const char *datasetName,
    DbscanExecType execType
);

/**
 * @brief Creates the benchmark file name for the DBSCAN algorithm.
 *
 * @param eps The epsilon value used in the DBSCAN algorithm.
 * @param minPts The minimum number of points required to form a cluster.
 * @return The benchmark file name, or nullptr if an error occurs.
 */
char *makeBenchmarkFileName(float eps, uint32_t minPts);

/**
 * @brief Lists all files in a folder, allocating memory dynamically.
 *
 * @param folderPath The path to the folder.
 * @param fileNames Pointer to dynamically allocated array of strings.
 * @param fileCount Pointer to where the number of files will be stored.
 * @return True if listing was successful, false otherwise.
 */
bool listFilesInFolder(const char *folderPath, char ***fileNames, uint32_t *fileCount);

/**
 * @brief Parses a CSV dataset file and extracts the points.
 *
 * @param filePath Input CSV file name.
 * @param x Pointer to where the x coordinates will be stored.
 * @param y Pointer to where the y coordinates will be stored.
 * @param n Pointer to where the number of points will be stored.
 * @return True if parsing was successful, false otherwise.
 */
bool parseDatasetFile(const char *filePath, float **x, float **y, uint32_t *n);

/**
 * @brief Writes points and cluster IDs to a CSV dbscan file.
 *
 * @param folderPath Folder path where the output file will be written.
 * @param fileName Output CSV file name.
 * @param x Pointer to the x coordinates.
 * @param y Pointer to the y coordinates.
 * @param cluster Array of cluster IDs for each point.
 * @param n Number of points.
 */
void writeDbscanFile(
    const char *folderPath,
    const char *fileName,
    const float *x,
    const float *y,
    const uint32_t *cluster,
    uint32_t n
);

/**
* @brief Writes benchmark data to a CSV file.
 *
 * @param folderPath The path to the folder where the output file will be written.
 * @param fileName The name of the benchmark output file.
 * @param datasetNames The array of dataset file names.
 * @param datasetNs The array of dataset point counts.
 * @param cpuTimes The array of CPU computation times.
 * @param gpuTimes The array of GPU computation times.
 * @param speedups The array of speedups.
 * @param benchmarkCount The number of benchmarks.
 */
void writeBenchmarkFile(
    const char *folderPath,
    const char *fileName,
    char * const *datasetNames,
    const uint32_t *datasetNs,
    const double *cpuTimes,
    const double *gpuTimes,
    const double *speedups,
    uint32_t benchmarkCount
);

#endif
