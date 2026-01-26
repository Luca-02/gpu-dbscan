#ifndef COMMON_H
#define COMMON_H

#define DATA_IN_PATH "../data_in/"
#define DATA_OUT_PATH "../data_out/"

#define INPUT_FILE DATA_IN_PATH "dataset_10000.csv"
#define OUTPUT_FILE_CPU DATA_OUT_PATH "dbscan_10000_cpu.csv"
#define OUTPUT_FILE_GPU DATA_OUT_PATH "dbscan_10000_gpu.csv"

#define EPSILON 0.3
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
 * @param x_min The minimum x coordinate of the grid.
 * @param y_min The minimum y coordinate of the grid.
 * @param eps The epsilon distance.
 */
HD inline void point_cell_coordinates(
    int *cx,
    int *cy,
    const double x,
    const double y,
    const double x_min,
    const double y_min,
    const double eps
) {
    *cx = (int) ((x - x_min) / eps);
    *cy = (int) ((y - y_min) / eps);
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
HD inline int cell_id(const int cx, const int cy, const int width) {
    return cy * width + cx;
}

/**
 * @brief Checks if a point is a core point based on its degree and minimum number of points.
 *
 * @param degree The degree of the point.
 * @param min_pts The minimum number of points.
 * @return True if the point is a core point, false otherwise.
 *
 * @note The point itself is counted in the degree.
 */
HD inline bool is_core(const int degree, const int min_pts) {
    return degree + 1 >= min_pts;
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
HD inline bool is_eps_neighbor(
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
