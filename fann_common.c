/* fann_common.c - Common implementations for FANN utilities */

#include "fann_common.h"

/* Activation function arrays */
const int SYM_FUNCTIONS[] = {
    FANN_LINEAR,
    FANN_GAUSSIAN_SYMMETRIC, FANN_COS_SYMMETRIC, FANN_SIN_SYMMETRIC,
    FANN_LINEAR_PIECE_SYMMETRIC, FANN_ELLIOT_SYMMETRIC,
    FANN_SIGMOID_SYMMETRIC_STEPWISE, FANN_SIGMOID_SYMMETRIC
};

const int MID_FUNCTIONS[] = {
    FANN_SIGMOID_STEPWISE, FANN_ELLIOT, FANN_LINEAR_PIECE,
    FANN_GAUSSIAN_STEPWISE, FANN_GAUSSIAN, FANN_COS, FANN_SIN, FANN_SIGMOID
};

const int IN_FUNCTIONS[] = {
    FANN_SIGMOID_SYMMETRIC, FANN_SIGMOID_SYMMETRIC_STEPWISE, FANN_GAUSSIAN_SYMMETRIC
};

const int OUT_FUNCTIONS[] = {
    FANN_GAUSSIAN_SYMMETRIC, FANN_SIGMOID_SYMMETRIC, FANN_SIGMOID_SYMMETRIC_STEPWISE
};

const size_t SYM_FUNCTIONS_SIZE = sizeof(SYM_FUNCTIONS) / sizeof(int);
const size_t MID_FUNCTIONS_SIZE = sizeof(MID_FUNCTIONS) / sizeof(int);
const size_t IN_FUNCTIONS_SIZE = sizeof(IN_FUNCTIONS) / sizeof(int);
const size_t OUT_FUNCTIONS_SIZE = sizeof(OUT_FUNCTIONS) / sizeof(int);

/* Rebuild activation functions for a specific layer */
void rebuild_functions_layer(struct fann *ann, int layer, int neurons) {
    int num_layers = fann_get_num_layers(ann);
    
    for (int neuron = 0; neuron < neurons; neuron++) {
        int func_idx;
        double steepness;
        
        if (layer == 1) {
            func_idx = MID_FUNCTIONS[rand() % MID_FUNCTIONS_SIZE];
        } else if (layer == num_layers - 1) {
            func_idx = OUT_FUNCTIONS[rand() % OUT_FUNCTIONS_SIZE];
        } else {
            func_idx = MID_FUNCTIONS[rand() % MID_FUNCTIONS_SIZE];
        }
        
        steepness = 0.1 + (rand() % 100) * 0.01;
        
        if (layer == num_layers - 1 || layer == 1) {
            steepness = 1.0f;
        }
        
        fann_set_activation_steepness(ann, steepness, layer, neuron);
        fann_set_activation_function(ann, func_idx, layer, neuron);
    }
}

/* Rebuild activation functions for all layers */
void rebuild_functions_all(struct fann *ann) {
    int num_layers = fann_get_num_layers(ann);
    
    printf("\\r\\n[ activation functions: ");
    for (int layer = 1; layer < num_layers; layer++) {
        int func_idx = rand() % (sizeof(enum fann_activationfunc_enum) * 4);
        if (func_idx == 1 || func_idx == 2) {
            func_idx = FANN_LINEAR_PIECE_SYMMETRIC;
        }
        
        double steepness = 0.1 + (rand() % 100) * 0.01;
        
        fann_set_activation_steepness_layer(ann, steepness, layer);
        fann_set_activation_function_layer(ann, func_idx, layer);
        
        printf("<layer#%02d %s:%-4.02f> ", layer, 
               FANN_ACTIVATIONFUNC_NAMES[fann_get_activation_function(ann, layer, 0)],
               fann_get_activation_steepness(ann, layer, 0));
    }
    printf("]\\r\\n");
}

/* Signal handler for graceful termination */
void sig_term(int p) {
    (void)p; /* Suppress unused parameter warning */
    printf("\\r\\nsaving network...\\r\\n");
    /* Network saving should be handled by calling code */
    exit(0);
}

/* Apply jitter to a value */
double jitter_value(double src_val, double jitter_factor) {
    double jitter = ((double)rand() / RAND_MAX - 0.5) * 2.0 * jitter_factor;
    return src_val + jitter;
}