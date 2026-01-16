#ifndef POINT_H
#define POINT_H

#define UNDEFINED (-2)
#define NOISE (-1)

typedef struct {
    double x;
    double y;
    int label;
} Point;

void init_point(Point *p, double x, double y);
double distance(const Point *a, const Point *b);

#endif