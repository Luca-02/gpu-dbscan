#ifndef IO_H
#define IO_H

/**
 * @brief Parses a CSV dataset file and extracts x and y coordinates.
 * @param filename Input CSV file name.
 * @param x Pointer to store the dynamically allocated x-coordinates array.
 * @param y Pointer to store the dynamically allocated y-coordinates array.
 * @param n Pointer to where the number of points will be stored.
 * @return Number of points parsed or -1 on error.
 */
bool parse_dataset_file(const char *filename, double **x, double **y, int *n);

/**
 * @brief Writes points and cluster IDs to a CSV dbscan file.
 * @param filename Output CSV file name.
 * @param x Array of x-coordinates.
 * @param y Array of y-coordinates.
 * @param cluster Array of cluster IDs for each point.
 * @param n Number of points.
 */
void write_dbscan_file(const char *filename, const double *x, const double *y, const int *cluster, int n);

#endif
