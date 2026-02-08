#include <cstdio>
#include <cstdlib>
#include <string>
#include <filesystem>
#include "./io.h"
#include "./common.h"
#include "./helper.h"

/**
 * @brief Compares two dataset file names by their dataset number.
 *
 * @param a Pointer to first file name.
 * @param b Pointer to second file name.
 * @return Negative if a < b, positive if a > b, zero if equal.
 */
int compareDatasetNames(const void *a, const void *b) {
    const char *fa = *(const char **) a;
    const char *fb = *(const char **) b;

    const char *pa = strstr(fa, "dataset");
    const char *pb = strstr(fb, "dataset");
    if (!pa || !pb) {
        return strcmp(fa, fb);
    }

    pa += 7;
    pb += 7;

    const int na = atoi(pa);
    const int nb = atoi(pb);

    if (na != nb) {
        return na - nb;
    }

    return strcmp(fa, fb);
}

/**
 * @brief Creates the output file name for the DBSCAN algorithm.
 *
 * @param datasetName The name of the input dataset file.
 * @param execType The execution type of the DBSCAN algorithm.
 * @return The output file name, or nullptr if an error occurs.
 */
char *makeDbscanOutputName(
    const char *datasetName,
    const DbscanExecType execType
) {
    if (!datasetName) return nullptr;

    // Take only the filename (remove the path)
    const char *filename = datasetName;
    const char *slash = strrchr(datasetName, '/');
    const char *backslash = strrchr(datasetName, '\\');
    if (slash || backslash) {
        const char *sep = slash > backslash ? slash : backslash;
        filename = sep + 1;
    }

    const char *datasetPrefix = "dataset";
    const size_t datasetPrefixLen = strlen(datasetPrefix);
    const bool isDatasetPrefix = strncmp(filename, datasetPrefix, datasetPrefixLen) == 0;

    const char *suffix = isDatasetPrefix ? filename + datasetPrefixLen : filename;
    const char *execStr = execType == DbscanExecType::CPU ? "cpu" : "gpu";
    const char *middle = "_dbscan";
    const size_t outLen = strlen(execStr) + strlen(middle) + strlen(suffix) + 1;

    char *outName = (char *) malloc(outLen);
    if (!outName) return nullptr;

    snprintf(outName, outLen, "%s%s%s", execStr, middle, suffix);

    return outName;
}

/**
 * @brief Creates the benchmark file name for the DBSCAN algorithm.
 *
 * @param eps The epsilon value used in the DBSCAN algorithm.
 * @param minPts The minimum number of points required to form a cluster.
 * @return The benchmark file name, or nullptr if an error occurs.
 */
char *makeBenchmarkFileName(const float eps, const uint32_t minPts) {
    char epsBuf[32];

    // Convert float to string
    snprintf(epsBuf, sizeof(epsBuf), "%f", eps);

    size_t len = strlen(epsBuf);

    // Remove trailing zeros
    while (len > 0 && epsBuf[len - 1] == '0') {
        epsBuf[--len] = '\0';
    }

    // Remove trailing dot if present
    if (len > 0 && epsBuf[len - 1] == '.') {
        epsBuf[--len] = '\0';
    }

    // Replace remaining '.' with 'd'
    for (char *p = epsBuf; *p; ++p) {
        if (*p == '.') {
            *p = 'd';
            break;
        }
    }

    const char *prefix = "benchmark_";
    const char *epsSuffix = "eps_";
    const char *ptsSuffix = "minpts";
    const char *ext = ".csv";

    // Calculate total length
    const size_t totalLen =
            strlen(prefix) +
            strlen(epsBuf) +
            strlen(epsSuffix) +
            10 + // max digits for uint32
            strlen(ptsSuffix) +
            strlen(ext) +
            1; // null terminator

    char *fileName = (char *) malloc(totalLen);
    if (!fileName) return nullptr;

    // Construct final string
    snprintf(fileName, totalLen, "%s%s%s%u%s%s",
             prefix,
             epsBuf,
             epsSuffix,
             minPts,
             ptsSuffix,
             ext
    );

    return fileName;
}

/**
 * @brief Lists all files in a folder, allocating memory dynamically.
 *
 * @param folderPath The path to the folder.
 * @param fileNames Pointer to dynamically allocated array of strings.
 * @param fileCount Pointer to where the number of files will be stored.
 *
 * @note Memory for fileNames array is dynamically allocated using malloc and must be freed by the caller.
 */
bool listFilesInFolder(const char *folderPath, char ***fileNames, uint32_t *fileCount) {
    *fileNames = nullptr;
    *fileCount = 0;

    if (!std::filesystem::exists(folderPath)) {
        fprintf(stderr, "Folder %s does not exist\n", folderPath);
        return false;
    }

    // First pass: count files
    for (const auto &entry: std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            (*fileCount)++;
        }
    }

    if (*fileCount == 0) {
        fprintf(stderr, "No files found in folder %s\n", folderPath);
        return false;
    }

    *fileNames = (char **) malloc(*fileCount * sizeof(char *));
    if (!*fileNames) return false;

    // Second pass: fill the array
    uint32_t idx = 0;
    for (const auto &entry: std::filesystem::directory_iterator(folderPath)) {
        if (!entry.is_regular_file()) continue;

        std::string fileName = entry.path().filename().string();
        (*fileNames)[idx] = (char *) malloc((fileName.size() + 1) * sizeof(char));
        if (!(*fileNames)[idx]) {
            return false;
        }
        strcpy((*fileNames)[idx], fileName.c_str());
        idx++;
    }

    return true;
}

/**
 * @brief Parses a CSV dataset file containing points in 2D space.
 * The CSV file is expected to have a header line, followed by lines containing
 * two floating-point numbers separated by a comma, representing x and y coordinates.
 *
 * @param filePath The path to the CSV dataset file.
 * @param x Pointer to where the x coordinates will be stored.
 * @param y Pointer to where the y coordinates will be stored.
 * @param n Pointer to where the number of points will be stored.
 * @return True if the file was parsed successfully, false otherwise.
 *
 * @note Memory for x and y array is dynamically allocated using malloc and must be freed by the caller.
 */
bool parseDatasetFile(const char *filePath, float **x, float **y, uint32_t *n) {
    *x = nullptr;
    *y = nullptr;
    *n = 0;

    char line[512];
    FILE *file = fopen(filePath, "r");
    if (!file) {
        fprintf(stderr, "Error opening dataset file %s\n", filePath);
        return false;
    }

    // Skip header
    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "Empty file or failed to read header\n");
        fclose(file);
        return false;
    }

    // First pass: count lines
    while (fgets(line, sizeof(line), file)) {
        (*n)++;
    }

    if (*n == 0) {
        fprintf(stderr, "No data found in file %s\n", filePath);
        fclose(file);
        return false;
    }

    rewind(file);
    fgets(line, sizeof(line), file); // skip header

    *x = (float *) malloc_s(*n * sizeof(float));
    *y = (float *) malloc_s(*n * sizeof(float));
    if (!*x || !*y) {
        fclose(file);
        return false;
    }

    // Second pass: fill the arrays
    uint32_t i = 0;
    while (fgets(line, sizeof(line), file) && i < *n) {
        char *end;

        (*x)[i] = strtof(line, &end);
        if (*end != ',') {
            fprintf(stderr, "Expected ',' after x at line %u\n", i + 2);
            fclose(file);
            return false;
        }

        // Skip comma
        end++;

        const char *start = end;
        (*y)[i] = strtof(start, &end);
        if (start == end) {
            fprintf(stderr, "Invalid y value at line %u\n", i + 1);
            fclose(file);
            return false;
        }

        i++;
    }

    fclose(file);
    return true;
}

/**
 * @brief Writes a CSV dbscan file containing points and their assigned cluster IDs.
 * The output file will have a header line "x,y,cluster" and then one line per point,
 * listing its x and y coordinates and the corresponding cluster ID.
 *
 * @param folderPath The path to the folder where the output file will be written.
 * @param fileName The dbscan output file name.
 * @param x Pointer to the array of x coordinates corresponding to each point.
 * @param y Pointer to the array of y coordinates corresponding to each point.
 * @param cluster Pointer to the array of cluster IDs corresponding to each point.
 * @param n The number of points to write.
 */
void writeDbscanFile(
    const char *folderPath,
    const char *fileName,
    const float *x,
    const float *y,
    const uint32_t *cluster,
    const uint32_t n
) {
    if (!std::filesystem::exists(folderPath)) {
        std::filesystem::create_directory(folderPath);
    }

    const std::string path = folderPath + std::string(fileName);

    FILE *file = fopen(path.c_str(), "w");
    if (!file) {
        fprintf(stderr, "Error opening dbscan output file %s\n", fileName);
        return;
    }

    // Write header
    fprintf(file, "x,y,cluster\n");

    for (uint32_t i = 0; i < n; i++) {
        fprintf(file, "%.7f,%.7f,%u\n", x[i], y[i], cluster[i]);
    }

    fclose(file);
}

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
    char *const *datasetNames,
    const uint32_t *datasetNs,
    const double *cpuTimes,
    const double *gpuTimes,
    const double *speedups,
    const uint32_t benchmarkCount
) {
    if (!std::filesystem::exists(folderPath)) {
        std::filesystem::create_directory(folderPath);
    }

    const std::string path = std::string(folderPath) + fileName;

    FILE *file = fopen(path.c_str(), "w");
    if (!file) {
        fprintf(stderr, "Error opening benchmark file\n");
        return;
    }

    // Write header
    fprintf(file, "dataset,n,cpu_time,gpu_time,speedup\n");

    for (uint32_t i = 0; i < benchmarkCount; i++) {
        fprintf(
            file, "%s,%u,%.7f,%.7f,%.7f\n",
            datasetNames[i], datasetNs[i], cpuTimes[i], gpuTimes[i], speedups[i]
        );
    }

    fclose(file);
}
