#include <cstdio>
#include <cstdlib>
#include "io.h"
#include "helper.h"

static void cleanup(FILE *file, double **x, double **y) {
    if (*x) {
        free(*x);
        *x = nullptr;
    }
    if (*y) {
        free(*y);
        *y = nullptr;
    }
    if (file) fclose(file);
}

/**
 * @brief Parses a CSV input file containing points in 2D space.
 * The CSV file is expected to have a header line, followed by lines containing
 * two floating-point numbers separated by a comma, representing x and y coordinates.
 *
 * @param filename The path to the CSV input file.
 * @param x Pointer to a double pointer where the x-coordinates array will be stored.
 * @param y Pointer to a double pointer where the y-coordinates array will be stored.
 * @param n Pointer to a size_t where the number of points will be stored.
 * @return The number of points read from the file on success, or -1 on error.
 *
 * @note Memory for x and y arrays is dynamically allocated using malloc and must be freed by the caller.
 */
bool parse_input_file(const char *filename, double **x, double **y, size_t *n) {
    *x = nullptr, *y = nullptr;
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

    size_t i = 0;
    while (fgets(line, sizeof(line), file) && i < *n) {
        char *end;

        (*x)[i] = strtod(line, &end);
        if (*end != ',') {
            fprintf(stderr, "Expected ',' after x at line %zu\n", i + 2);
            cleanup(file, x, y);
            return false;
        }

        // Skip comma
        end++;

        const char *start = end;
        (*y)[i] = strtod(start, &end);
        if (start == end) {
            fprintf(stderr, "Invalid y value at line %zu\n", i + 1);
            cleanup(file, x, y);
            return false;
        }

        i++;
    }

    fclose(file);
    return true;
}

/**
 * @brief Writes a CSV output file containing points and their assigned cluster IDs.
 * The output file will have a header line "x,y,cluster" and then one line per point,
 * listing its x and y coordinates and the corresponding cluster ID.
 *
 * @param filename The path to the CSV output file.
 * @param x Pointer to the array of x-coordinates.
 * @param y Pointer to the array of y-coordinates.
 * @param cluster Pointer to the array of cluster IDs corresponding to each point.
 * @param n The number of points to write.
 *
 * @note The arrays x, y, and cluster must have at least n elements.
 */
void write_output_file(const char *filename, const double *x, const double *y, const int *cluster, const size_t n) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }

    // Write header
    fprintf(file, "x,y,cluster\n");

    for (size_t i = 0; i < n; i++) {
        fprintf(file, "%.15g,%.15g,%d\n", x[i], y[i], cluster[i]);
    }

    fclose(file);
}
