#ifndef COMMON_H
#define COMMON_H

#define DATA_IN_PATH "../data_in/"
#define DATA_OUT_PATH "../data_out/"

#define INPUT_FILE DATA_IN_PATH "dataset_1000000n_30c_100d0cs_3d0std_0d001nr.csv"
#define OUTPUT_FILE_CPU DATA_OUT_PATH "cpu.csv"
#define OUTPUT_FILE_GPU DATA_OUT_PATH "gpu.csv"

#define EPSILON 3.0
#define MIN_PTS 8
#define NO_CLUSTER_LABEL 0

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

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
    const double x,
    const double y,
    const double xMin,
    const double yMin,
    const double invEps
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
    const double x1,
    const double y1,
    const double x2,
    const double y2,
    const double eps2
) {
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    return dx * dx + dy * dy <= eps2;
}

#endif
