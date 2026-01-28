#ifndef IO_H
#define IO_H

/**
 * @brief Parses a CSV dataset file and extracts the points.
 * @param filename Input CSV file name.
 * @param x Pointer to where the x coordinates will be stored.
 * @param y Pointer to where the y coordinates will be stored.
 * @param n Pointer to where the number of points will be stored.
 * @return True if parsing was successful, false otherwise.
 */
bool parseDatasetFile(const char *filename, double **x, double **y, int *n);

/**
 * @brief Writes points and cluster IDs to a CSV dbscan file.
 * @param filename Output CSV file name.
 * @param x Pointer to the x coordinates.
 * @param y Pointer to the y coordinates.
 * @param cluster Array of cluster IDs for each point.
 * @param n Number of points.
 */
void writeDbscanFile(const char *filename, const double *x, const double *y, const int *cluster, int n);

#endif
