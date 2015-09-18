//#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include "fann/doublefann.h"

int main(int argc,char*argv[])
{
		struct fann *ffann=NULL,*good_network=NULL;
    
		if(argc<=1)
			exit(48);
    if ( ( ffann = fann_create_from_file ( argv[1] ) ) !=NULL )
    {
        printf ( "network %p. nettype: %u conn_rate: %.2f layers: %u connections: %u neur: %u\r\n",ffann ,fann_get_network_type(ffann),
                 fann_get_connection_rate(ffann),
                 fann_get_num_layers(ffann),
                 fann_get_total_connections(ffann),
                 fann_get_total_neurons(ffann));
        fann_print_connections(ffann);

    }  	
    else
    {

    //    ann = fann_create_from_file ( "train.net" );
        printf("следууюющий!",ffann);
        //return 1;
    }
    
	return(2);
}