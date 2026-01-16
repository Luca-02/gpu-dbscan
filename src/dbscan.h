#ifndef DBSCAN_H
#define DBSCAN_H

#include "structure/point.h"

typedef struct {
    int n;
    int *cum_deg;
    int *adj;
} Graph;

void free_graph(Graph *graph);
void build_graph(Graph *graph, const Point *points, int n, double eps);
void dbscan(Point *points, int n, double eps, int min_pts);
void write_output(const char *filename, const Point *points, int n);

#endif