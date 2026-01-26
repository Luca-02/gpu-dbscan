#ifndef IO_H
#define IO_H

/**
 * @brief Parses a CSV dataset file and extracts the points.
 * @param filename Input CSV file name.
 * @param points Pointer to store the flattened 2D points array.
 * @param n Pointer to where the number of points will be stored.
 * @return True if parsing was successful, false otherwise.
 */
bool parse_dataset_file(const char *filename, double **points, int *n);

/**
 * @brief Writes points and cluster IDs to a CSV dbscan file.
 * @param filename Output CSV file name.
 * @param points Flattened array of 2D points.
 * @param cluster Array of cluster IDs for each point.
 * @param n Number of points.
 */
void write_dbscan_file(const char *filename, const double *points, const int *cluster, int n);

#endif
