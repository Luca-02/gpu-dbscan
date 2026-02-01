#ifndef IO_H
#define IO_H

#define MAX_FILES 1024

/**
 * @brief Lists all files in a folder, allocating memory dynamically.
 *
 * @param folderPath The path to the folder.
 * @param fileNames Pointer to dynamically allocated array of strings.
 * @param fileCount Pointer to where the number of files will be stored.
 * @return True if listing was successful, false otherwise.
 */
bool listFilesInFolder(const char *folderPath, char ***fileNames, uint32_t *fileCount);

/**
 * @brief Parses a CSV dataset file and extracts the points.
 *
 * @param fileName Input CSV file name.
 * @param x Pointer to where the x coordinates will be stored.
 * @param y Pointer to where the y coordinates will be stored.
 * @param n Pointer to where the number of points will be stored.
 * @return True if parsing was successful, false otherwise.
 */
bool parseDatasetFile(const char *fileName, float **x, float **y, uint32_t *n);

/**
 * @brief Writes points and cluster IDs to a CSV dbscan file.
 *
 * @param fileName Output CSV file name.
 * @param x Pointer to the x coordinates.
 * @param y Pointer to the y coordinates.
 * @param cluster Array of cluster IDs for each point.
 * @param n Number of points.
 */
void writeDbscanFile(const char *fileName, const float *x, const float *y, const uint32_t *cluster, uint32_t n);

#endif

