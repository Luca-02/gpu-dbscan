#ifndef DBSCAN_H
#define DBSCAN_H

#define INPUT_FILE "../data/input.csv"
#define OUTPUT_FILE "../data/output.csv"

/**
 * @brief The maximum Euclidean distance between two points to consider them neighbors.
 */
#define EPSILON 5
/**
 * @brief The minimum number of points to form a dense region.
 */
#define MIN_PTS 30

/**
 * @brief A point that has not yet been processed.
 */
#define UNDEFINED (-2)
/**
 * @brief A point with less than MIN_PTS neighbors and is not part of a dense region.
 */
#define NOISE (-1)

#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

/**
 * @brief Checks if a point is a core point based on its degree and minimum number of points.
 *
 * @param deg The degree of the point.
 * @param min_pts The minimum number of points.
 * @return True if the point is a core point, false otherwise.
 *
 * @note The point itself is counted in the degree.
 */
HD inline bool is_core(const size_t deg, const size_t min_pts) {
    return deg >= min_pts;
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
HD inline bool is_eps_neighbor(const double x1, const double y1, const double x2, const double y2, const double eps) {
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    return dx * dx + dy * dy <= eps * eps;
}

void dbscan_cpu(int *cluster, const double *x, const double *y, size_t n, double eps, size_t min_pts);

void dbscan_gpu(int *cluster, const double *x, const double *y, size_t n, double eps, size_t min_pts);

#endif
