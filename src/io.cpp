#include <cstdio>
#include <cstdlib>
#include "io.h"

int parse_input_file(const char *filename, double **x, double **y) {
    FILE *file;
    if (fopen_s(&file, filename, "r") != 0) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return -1;
    }

    int points_count = 0;

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        points_count++;
    }

    printf("Parsed %d points\n", points_count);
    rewind(file);

    *x = (double *) malloc(points_count * sizeof(double));
    *y = (double *) malloc(points_count * sizeof(double));

    int i = 0;
    while (fgets(line, sizeof(line), file) && i < points_count) {
        char *end_ptr;

        (*x)[i] = strtod(line, &end_ptr);
        if (end_ptr == line) {
            fprintf(stderr, "Invalid x value at line %d\n", i + 1);
            fclose(file);
            return -1;
        }

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

void write_output_file(const char *filename, const double *x, const double *y, const int *cluster, const int n) {
    FILE *file;
    if (fopen_s(&file, filename, "w") != 0) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }

    for (int i = 0; i < n; i++) {
        fprintf(file, "%lf %lf %d\n", x[i], y[i], cluster[i]);
    }

    fclose(file);
}
