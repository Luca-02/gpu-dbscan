#include <cstdio>
#include <cstdlib>
#include <string>
#include <filesystem>
#include "io.h"
#include "common.h"
#include "helper.h"

/**
 * @brief Parses a CSV dataset file containing points in 2D space.
 * The CSV file is expected to have a header line, followed by lines containing
 * two floating-point numbers separated by a comma, representing x and y coordinates.
 *
 * @param fileName The path to the CSV dataset file.
 * @param x Pointer to where the x coordinates will be stored.
 * @param y Pointer to where the y coordinates will be stored.
 * @param n Pointer to where the number of points will be stored.
 * @return True if the file was parsed successfully, false otherwise.
 *
 * @note Memory for x and y array is dynamically allocated using malloc and must be freed by the caller.
 */
bool parseDatasetFile(const char *fileName, double **x, double **y, int *n) {
    *x = nullptr;
    *y = nullptr;
    *n = 0;

    char line[512];
    FILE *file = fopen(fileName, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", fileName);
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
        fprintf(stderr, "No data found in file %s\n", fileName);
        fclose(file);
        return false;
    }

    rewind(file);
    fgets(line, sizeof(line), file); // skip header

    *x = (double *) malloc_s(*n * sizeof(double));
    *y = (double *) malloc_s(*n * sizeof(double));
    if (!*x || !*y) {
        fclose(file);
        return false;
    }

    // Second pass: fill the arrays
    int i = 0;
    while (fgets(line, sizeof(line), file) && i < *n) {
        char *end;

        (*x)[i] = strtod(line, &end);
        if (*end != ',') {
            fprintf(stderr, "Expected ',' after x at line %u\n", i + 2);
            fclose(file);
            return false;
        }

        // Skip comma
        end++;

        const char *start = end;
        (*y)[i] = strtod(start, &end);
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
 * @param fileName The path to the CSV output file.
 * @param x Pointer to the array of x coordinates corresponding to each point.
 * @param y Pointer to the array of y coordinates corresponding to each point.
 * @param cluster Pointer to the array of cluster IDs corresponding to each point.
 * @param n The number of points to write.
 *
 * @note The array points and cluster must have [n * 2] and [n] elements.
 */
void writeDbscanFile(const char *fileName, const double *x, const double *y, const int *cluster, const int n) {
    if (!std::filesystem::exists(DATA_OUT_PATH)) {
        std::filesystem::create_directory(DATA_OUT_PATH);
    }

    FILE *file = fopen(fileName, "w");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", fileName);
        return;
    }

    // Write header
    fprintf(file, "x,y,cluster\n");

    for (int i = 0; i < n; i++) {
        fprintf(file, "%.15g,%.15g,%u\n", x[i], y[i], cluster[i]);
    }

    fclose(file);
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
bool listFilesInFolder(const char *folderPath, char ***fileNames, int *fileCount) {
    *fileNames = nullptr;
    *fileCount = 0;

    if (!std::filesystem::exists(folderPath)) {
        fprintf(stderr, "Folder %s does not exist\n", folderPath);
        return false;
    }

    // First pass: count files
    for (const auto &entry : std::filesystem::directory_iterator(folderPath)) {
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
    int idx = 0;
    for (const auto &entry : std::filesystem::directory_iterator(folderPath)) {
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