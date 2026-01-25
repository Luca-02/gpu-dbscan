#ifndef COMMON_H
#define COMMON_H

#define DATA_PATH "../data/"

#define INPUT_FILE DATA_PATH "dataset_1000000.csv"
#define OUTPUT_FILE_CPU DATA_PATH "dbscan_1000000_cpu.csv"
#define OUTPUT_FILE_GPU DATA_PATH "dbscan_1000000_gpu.csv"

/**
 * @brief The maximum Euclidean distance between two points to consider them neighbors.
 */
#define EPSILON(n) (0.3 * sqrt(n))
/**
 * @brief The minimum number of points to form a dense region.
 */
#define MIN_PTS 8

/**
 * @brief Default cluster id.
 */
#define NO_CLUSTER 0

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
HD inline void cell_coordinates(
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
 * @param eps The epsilon distance.
 * @return True if the points are epsilon neighbors, false otherwise.
 *
 * @note This implementation avoids the use of sqrt by checking the squared distance instead.
 */
HD inline bool is_eps_neighbor(
    const double x1,
    const double y1,
    const double x2,
    const double y2,
    const double eps
) {
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    return dx * dx + dy * dy <= eps * eps;
}

#endif
