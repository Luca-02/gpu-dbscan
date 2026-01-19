#include <cstdio>
#include <ctime>
#include "io.h"
#include "dbscan.h"


int main() {
    double *x, *y;
    const int n = parse_input_file(INPUT_FILE, &x, &y);

    if (n < 0) {
        fprintf(stderr, "Error parsing points file\n");
        return -1;
    }

    int *cluster = (int *) malloc(n * sizeof(double));
    const clock_t start = clock();
    dbscan_cpu(cluster, x, y, n, EPSILON, MIN_PTS);
    const clock_t end = clock();

    const double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Sequential DBSCAN elapsed time: %f seconds", elapsed);

    write_output_file(OUTPUT_FILE, x, y, cluster, n);

    free(x);
    free(y);
    free(cluster);
    return 0;
}
