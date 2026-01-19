#include <cstdio>
#include <cstdlib>
#include "io.h"


/**
 * @brief Parses a CSV input file containing points in 2D space.
 * The CSV file is expected to have a header line, followed by lines containing
 * two floating-point numbers separated by a comma, representing x and y coordinates.
 *
 * @param filename The path to the CSV input file.
 * @param x Pointer to a double pointer where the x-coordinates array will be stored.
 * @param y Pointer to a double pointer where the y-coordinates array will be stored.
 * @return The number of points read from the file on success, or -1 on error.
 *
 * @note Memory for x and y arrays is dynamically allocated using malloc and must be freed by the caller.
 */
int parse_input_file(const char *filename, double **x, double **y) {
    FILE *file;
    if (fopen_s(&file, filename, "r") != 0) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return -1;
    }

    int points_count = 0;
    char line[256];

    // Skip header
    if (!fgets(line, sizeof(line), file)) {
        fprintf(stderr, "Empty file or failed to read header\n");
        fclose(file);
        return -1;
    }

    while (fgets(line, sizeof(line), file)) {
        points_count++;
    }

    printf("Parsed %d points\n", points_count);
    rewind(file);
    // Skip header
    fgets(line, sizeof(line), file);

    *x = (double *) malloc(points_count * sizeof(double));
    *y = (double *) malloc(points_count * sizeof(double));

    int i = 0;
    while (fgets(line, sizeof(line), file) && i < points_count) {
        char *end_ptr;

        (*x)[i] = strtod(line, &end_ptr);
        if (*end_ptr != ',') {
            fprintf(stderr, "Expected ',' after x at line %d\n", i + 2);
            fclose(file);
            return -1;
        }

        // Skip comma
        end_ptr++;

        (*y)[i] = strtod(end_ptr, &end_ptr);
        if (end_ptr == line) {
            fprintf(stderr, "Invalid y value at line %d\n", i + 1);
            fclose(file);
            return -1;
        }

        i++;
    }

    fclose(file);
    return points_count;
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
void write_output_file(const char *filename, const double *x, const double *y, const int *cluster, const int n) {
    FILE *file;
    if (fopen_s(&file, filename, "w") != 0) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }

    // Write header
    fprintf(file, "x,y,cluster\n");

    for (int i = 0; i < n; i++) {
        fprintf(file, "%.15g,%.15g,%d\n", x[i], y[i], cluster[i]);
    }

    fclose(file);
}
