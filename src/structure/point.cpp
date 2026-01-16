#include <cmath>
#include "point.h"

void init_point(Point *p, const double x, const double y) {
    p->x = x;
    p->y = y;
    p->label = UNDEFINED;
}

double distance(const Point *a, const Point *b) {
    return sqrt(
        (a->x - b->x) * (a->x - b->x) +
        (a->y - b->y) * (a->y - b->y)
    );
}
