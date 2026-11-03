/* Optional phase timers for the axis loops (speed work, not upstream).
 * Set PYWT_TIMING=1: totals per phase are printed to stderr at exit. Not thread safe. */
#ifndef PYWT_TIMING_H_
#define PYWT_TIMING_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum {T_DEC, T_REC};                    /* dwt (downcoef_axis), idwt (idwt_axis) */
enum {T_IN, T_KERNEL, T_OUT};           /* copy strided line in, convolution, copy line out */

static double pywt_t[2][3];
static long pywt_calls[2];
static int pywt_timing = -1;

static void pywt_timing_report(void){
    const char *name[2] = {"dwt ", "idwt"};
    for (int k = 0; k < 2; ++k)
        fprintf(stderr, "pywt timing %s (%4ld calls): copy in %6.3f s   kernel %6.3f s   copy out %6.3f s\n",
                name[k], pywt_calls[k], pywt_t[k][T_IN], pywt_t[k][T_KERNEL], pywt_t[k][T_OUT]);
}

static inline int pywt_timing_on(void){
    if (pywt_timing < 0){
        pywt_timing = getenv("PYWT_TIMING") != NULL;
        if (pywt_timing)
            atexit(pywt_timing_report);
    }
    return pywt_timing;
}

static inline double pywt_now(void){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

#endif
