#ifndef IO_H
#define IO_H

int parse_input_file(const char *filename, double **x, double **y);

void write_output_file(const char *filename, const double *x, const double *y, const int *cluster, int n);

#endif
