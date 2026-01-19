#include <cstdio>
#include <ctime>
#include "helper.h"
#include "io.h"
#include "dbscan.h"


int main() {
    double *x, *y;
    size_t n;

    if (!parse_input_file(INPUT_FILE, &x, &y, &n)) {
        fprintf(stderr, "Error parsing points file\n");
        return EXIT_FAILURE;
    }

    int *cluster = (int *) malloc_s(n * sizeof(int));
    if (!cluster) {
        free(x);
        free(y);
        return EXIT_FAILURE;
    }

    const clock_t start = clock();
    dbscan_cpu(cluster, x, y, n, EPSILON, MIN_PTS);
    const clock_t end = clock();

    const double elapsed = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Sequential DBSCAN elapsed time: %f seconds\n", elapsed);

    write_output_file(OUTPUT_FILE, x, y, cluster, n);

    free(x);
    free(y);
    free(cluster);
    return EXIT_SUCCESS;
}
