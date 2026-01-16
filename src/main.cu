#include <cstdio>
#include "parser.h"
#include "dbscan.h"

#define EPSILON 0.5
#define MIN_PTS 8

int main() {
    Point *points;
    const int n = parse_points_file("../data/input.txt", &points);

    if (n < 0) {
        fprintf(stderr, "Error parsing points file\n");
        return -1;
    }

    dbscan(points, n, EPSILON, MIN_PTS);
    write_output("../data/output.txt", points, n);

    free(points);
    return 0;
}
