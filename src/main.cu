#include <cstdio>
#include "parser.h"
#include "dbscan.h"

#define EPSILON 5
#define MIN_PTS 30

#define INPUT_FILE "../data/input.txt"
#define OUTPUT_FILE "../data/output.txt"

int main() {
    Point *points;
    const int n = parse_points_file(INPUT_FILE, &points);

    if (n < 0) {
        fprintf(stderr, "Error parsing points file\n");
        return -1;
    }

    dbscan(points, n, EPSILON, MIN_PTS);
    write_output(OUTPUT_FILE, points, n);

    free(points);
    return 0;
}
