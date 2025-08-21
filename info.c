/* Network Information Utility
 * Displays detailed information about FANN network files
 */

#include <stdlib.h>
#include <stdio.h>
#include "fann_common.h"

void print_usage(const char *progname) {
    printf("Usage: %s <network_file.net>\n", progname);
    printf("Displays information about a FANN neural network file\n");
}

int main(int argc, char *argv[]) {
    struct fann *network = NULL;
    
    if (argc <= 1) {
        print_usage(argv[0]);
        exit(1);
    }
    
    network = fann_create_from_file(argv[1]);
    if (network != NULL) {
        printf("Network file: %s\n", argv[1]);
        printf("Network type: %u\n", fann_get_network_type(network));
        printf("Connection rate: %.2f\n", fann_get_connection_rate(network));
        printf("Layers: %u\n", fann_get_num_layers(network));
        printf("Total connections: %u\n", fann_get_total_connections(network));
        printf("Total neurons: %u\n", fann_get_total_neurons(network));
        
        printf("\nNetwork connections:\n");
        fann_print_connections(network);
        
        fann_destroy(network);
    } else {
        printf("Error: Could not load network from %s\n", argv[1]);
        return 1;
    }
    
    return 0;
}