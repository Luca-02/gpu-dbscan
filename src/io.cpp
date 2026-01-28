#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include "io.h"
#include "common.h"
#include "helper.h"

/**
 * @brief Cleans up resources used for IO.
 *
 * @param file The file pointer to close.
 * @param x The double pointer to free.
 * @param y The double pointer to free.
 */
static void cleanup(FILE *file, double **x, double **y) {
    if (file) fclose(file);
    if (*x) free(*x);
    if (*y) free(*y);
    *x = nullptr;
    *y = nullptr;
}

/**
 * @brief Parses a CSV dataset file containing points in 2D space.
 * The CSV file is expected to have a header line, followed by lines containing
 * two floating-point numbers separated by a comma, representing x and y coordinates.
 *
 * @param filename The path to the CSV dataset file.
 * @param x Pointer to where the x coordinates will be stored.
 * @param y Pointer to where the y coordinates will be stored.
 * @param n Pointer to where the number of points will be stored.
 * @return True if the file was parsed successfully, false otherwise.
 *
 * @note Memory for points array is dynamically allocated using malloc and must be freed by the caller.
 */
bool parseDatasetFile(const char *filename, double **x, double **y, int *n) {
    *x = nullptr;
    *y = nullptr;
    *n = 0;

    char line[512];
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return false;
    }

    // Skip header
    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "Empty file or failed to read header\n");
        cleanup(file, x, y);
        return false;
    }

    // Count lines
    while (fgets(line, sizeof(line), file)) {
        (*n)++;
    }

    if (*n == 0) {
        fprintf(stderr, "No data found in file %s\n", filename);
        cleanup(file, x, y);
        return false;
    }

    rewind(file);
    fgets(line, sizeof(line), file); // skip header

    *x = (double *) malloc_s(*n * sizeof(double));
    *y = (double *) malloc_s(*n * sizeof(double));
    if (!*x || !*y) {
        cleanup(file, x, y);
        return false;
    }

    int i = 0;
    while (fgets(line, sizeof(line), file) && i < *n) {
        char *end;

        (*x)[i] = strtod(line, &end);
        if (*end != ',') {
            fprintf(stderr, "Expected ',' after x at line %u\n", i + 2);
            cleanup(file, x, y);
            return false;
        }

        // Skip comma
        end++;

        const char *start = end;
        (*y)[i] = strtod(start, &end);
        if (start == end) {
            fprintf(stderr, "Invalid y value at line %u\n", i + 1);
            cleanup(file, x, y);
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
 * @param filename The path to the CSV output file.
 * @param x Pointer to the array of x coordinates corresponding to each point.
 * @param y Pointer to the array of y coordinates corresponding to each point.
 * @param cluster Pointer to the array of cluster IDs corresponding to each point.
 * @param n The number of points to write.
 *
 * @note The array points and cluster must have [n * 2] and [n] elements.
 */
void writeDbscanFile(const char *filename, const double *x, const double *y, const int *cluster, const int n) {
    if (!std::filesystem::exists(DATA_OUT_PATH)) {
        std::filesystem::create_directory(DATA_OUT_PATH);
    }

    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }

    // Write header
    fprintf(file, "x,y,cluster\n");

    for (int i = 0; i < n; i++) {
        fprintf(file, "%.15g,%.15g,%u\n", x[i], y[i], cluster[i]);
    }

    fclose(file);
}
