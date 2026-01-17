#ifndef CPU_DBSCAN_H
#define CPU_DBSCAN_H

inline int is_core(const int deg, const int min_pts) {
    // +1 because the point itself its also included
    return deg + 1 >= min_pts;
}

inline bool is_eps_neighbor(const double ax, const double ay, const double bx, const double by, const double eps) {
    // Avoid sqrt for better performance
    return (ax - bx) * (ax - bx) + (ay - by) * (ay - by) <= eps * eps;
}

void dbscan(int *cluster, const double *x, const double *y, int n, double eps, int min_pts);

#endif
