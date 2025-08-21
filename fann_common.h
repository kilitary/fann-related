/* fann_common.h - Common definitions and functions for FANN utilities */

#ifndef FANN_COMMON_H
#define FANN_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <memory.h>
#include <time.h>
#include <doublefann.h>

/* Common macros */
#define max(a,b) ((a>b) ? a : b)
#define min(a,b) ((a<b) ? a : b)

/* Common activation function arrays */
extern const int SYM_FUNCTIONS[];
extern const int MID_FUNCTIONS[];
extern const int IN_FUNCTIONS[];
extern const int OUT_FUNCTIONS[];
extern const size_t SYM_FUNCTIONS_SIZE;
extern const size_t MID_FUNCTIONS_SIZE;
extern const size_t IN_FUNCTIONS_SIZE;
extern const size_t OUT_FUNCTIONS_SIZE;

/* Common function declarations */
void rebuild_functions_layer(struct fann *ann, int layer, int neurons);
void rebuild_functions_all(struct fann *ann);
void sig_term(int p);
double jitter_value(double src_val, double jitter_factor);

#endif /* FANN_COMMON_H */