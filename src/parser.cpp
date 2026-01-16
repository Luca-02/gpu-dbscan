#include <cstdio>
#include <cstdlib>
#include "parser.h"

int parse_points_file(const char *filename, Point **points) {
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

    *points = (Point *) malloc(points_count * sizeof(Point));

    int i = 0;
    while (fgets(line, sizeof(line), file) && i < points_count) {
        char *end_ptr;

        const double x = strtod(line, &end_ptr);
        if (end_ptr == line) {
            fprintf(stderr, "Invalid x value at line %d\n", i + 1);
            fclose(file);
            return -1;
        }

        const double y = strtod(end_ptr, &end_ptr);
        if (end_ptr == line) {
            fprintf(stderr, "Invalid y value at line %d\n", i + 1);
            fclose(file);
            return -1;
        }

        init_point(&(*points)[i], x, y);
        i++;
    }

    fclose(file);
    return points_count;
}
