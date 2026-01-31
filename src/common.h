#ifndef COMMON_H
#define COMMON_H

#define DATA_IN_PATH "../data_in/"
#define DATA_OUT_PATH "../data_out/"

#define TEST_INPUT_DATASET DATA_IN_PATH "dataset20_1000000n_30c_1d0cs_0d03std_0d001nr.csv"
#define TEST_OUTPUT_DBSCAN_CPU DATA_OUT_PATH "cpu.csv"
#define TEST_OUTPUT_DBSCAN_GPU DATA_OUT_PATH "gpu.csv"

#define EPSILON 0.03
#define MIN_PTS 8
#define NO_CLUSTER_LABEL 0

#ifdef __CUDACC__
/** Macro to define host and device functions */
#define HD __host__ __device__
#else
#define HD // Empty definition for CPU compilation
#endif

/**
 * @brief Compares two dataset file names by their dataset number.
 *
 * @param a Pointer to first file name.
 * @param b Pointer to second file name.
 * @return Negative if a < b, positive if a > b, zero if equal.
 */
inline int compareDatasetNames(const void *a, const void *b) {
    const char *fa = *(const char **)a;
    const char *fb = *(const char **)b;

    const char *pa = strstr(fa, "dataset");
    const char *pb = strstr(fb, "dataset");
    if (!pa || !pb) {
        return strcmp(fa, fb);
    }

    pa += 7;
    pb += 7;

    const int na = atoi(pa);
    const int nb = atoi(pb);

    if (na != nb) {
        return na - nb;
    }

    return strcmp(fa, fb);
}

/**
 * @brief Calculates the cell coordinates of a point based on its coordinates and the epsilon
 * distance, that is the cell size.
 *
 * @param cx Pointer to store the calculated x coordinate of the cell.
 * @param cy Pointer to store the calculated y coordinate of the cell.
 * @param x The x coordinate of the point.
 * @param y The y coordinate of the point.
 * @param xMin The minimum x coordinate of the grid.
 * @param yMin The minimum y coordinate of the grid.
 * @param invEps The inverse of the epsilon distance.
 */
HD inline void pointCellCoordinates(
    int *cx,
    int *cy,
    const float x,
    const float y,
    const float xMin,
    const float yMin,
    const float invEps
) {
    *cx = (int) ((x - xMin) * invEps);
    *cy = (int) ((y - yMin) * invEps);
}

/**
 * @brief Calculates the cell ID of a point with linearization based on its cell coordinates
 * and the width of the grid.
 *
 * @param cx The x coordinate of the cell.
 * @param cy The y coordinate of the cell.
 * @param width The width of the grid.
 * @return The cell ID.
 */
HD inline int linearCellId(const int cx, const int cy, const int width) {
    return cy * width + cx;
}

/**
 * @brief Checks if a point is a core point based on its degree and minimum number of points.
 *
 * @param degree The degree of the point.
 * @param minPts The minimum number of points.
 * @return True if the point is a core point, false otherwise.
 *
 * @note The point itself is counted in the degree.
 */
HD inline bool isCore(const int degree, const int minPts) {
    return degree + 1 >= minPts;
}

/**
 * @brief Checks if two points are epsilon neighbors based on their coordinates and the epsilon distance.
 *
 * @param x1 The x coordinate of the first point.
 * @param y1 The y coordinate of the first point.
 * @param x2 The x coordinate of the second point.
 * @param y2 The y coordinate of the second point.
 * @param eps2 The epsilon distance squared.
 * @return True if the points are epsilon neighbors, false otherwise.
 *
 * @note This implementation avoids the use of sqrt by checking the squared distance instead.
 */
HD inline bool isEpsNeighbor(
    const float x1,
    const float y1,
    const float x2,
    const float y2,
    const float eps2
) {
    const float dx = x2 - x1;
    const float dy = y2 - y1;
    return dx * dx + dy * dy <= eps2;
}

#endif
