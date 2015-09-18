#include <stdio.h>
#include <signal.h>
#include <time.h>
#include <string.h>
#include <fann/doublefann.h>

#define max(a,b) ((a>b) ? a : b)
#define min(a,b) ((a<b) ? a : b)

unsigned int num_layers = 3;
unsigned int num_neurons_hidden = 256;
double desired_error = (  double ) 0.0001f;
unsigned int max_epochs = 1500000;
unsigned int epochs_between_reports = 175;
struct fann *ann;
struct fann_train_data *train_data, *test_data;
double mse_train, mse_test,prev_mse, min_mse_train=1, min_mse_test=1;
unsigned int i = 0,last_bads=0;
unsigned int bit_fail_train, bit_fail_test;
int lowest_test_mse_epoch=0;
int nextalgo=0;
int func_num=0;
double stagn_epoch=0;
int prevbitfail=0;
double prevsarep=0,prev_mse_test=0;
double stpns;
unsigned stpns_epoch=0;
double mse_chg=0;
unsigned lastmsechecktime=0;
double minutes_left=0;
double prev_mse_chg[61];
int cur_mse_chg=0;
double last_min_timeleft;
unsigned last_min_timeleft_upd=0;
double weight_mse;
struct fann_train_data *weight_data,*cln_test_data,*cln_weight_data,*cln_train_data;
int l1n=0,l2n=0,l3n=0,l4n=0,l5n=0,l6n=0;
int numn=3;
double conn_rate=1.0f;
int finaldatanum;
int reject_total=0;
int num,u;
int classmin=0;
struct fann_train_data * final_data,*final_test_data;
unsigned train_classes_added[10];
void rebuild_functions(void);
unsigned train_pos = 0;
unsigned finaltestdatanum=0;
unsigned *train_matrix;


void sig_term ( int p )
{
    printf ( "\r\nsaving net...\r\n" );
    fann_save ( ann, "bb-normal.net" );
    exit ( 0 );
}

void 	train_func( unsigned int num, unsigned int numinp, unsigned int numout, fann_type * input, fann_type * output)
{


    int addthis;
    int i;
    int added=0;

    if (train_pos>fann_length_train_data(weight_data)||num>fann_length_train_data(weight_data))
    {
        printf("err");
        return;
    }

    while (!added)
    {
        addthis=1;
        for ( i=0;i<weight_data->num_output;i++)
        {

            if (weight_data->output[train_pos][i]==1 && train_classes_added[i]++>=classmin)
            {

                addthis=0;
                break;
            }
            //printf("%d\r\n",train_classes_added[i]);
        }

        if (!addthis)
        {
            //	fprintf(stderr,"x");
            train_matrix[train_pos]=1;
            train_pos++;
            finaltestdatanum++;

            continue;
        }

        //	fprintf(stderr,".");
        train_matrix[train_pos]=0;
        int y;
        for (y=0;y<weight_data->num_input;y++)
        {
            input[y]=weight_data->input[train_pos][y];
        }
        for (y=0;y<weight_data->num_output;y++)
        {
            if (weight_data->output[train_pos][y])
                output[y]=weight_data->output[train_pos][y];
            else
                output[y]=0;
        }
        added=1;
        train_pos++;
    }

}



void 	test_train_func( unsigned int num, unsigned int numinp, unsigned int numout, fann_type * input, fann_type * output)
{

    if (num>finaldatanum)
        return;

    int addthis;
    int i;
    int added=0;

    while (!added)
    {
        addthis=1;

        if (!train_matrix[train_pos])
        {
            //	printf("x");
            train_pos++;
            //	finaltestdatanum++;
            continue;
        }

        //printf(".");

        int y;
        for (y=0;y<weight_data->num_input;y++)
        {
            input[y]=weight_data->input[train_pos][y];
        }
        for (y=0;y<weight_data->num_output;y++)
        {
            if (weight_data->output[train_pos][y])
                output[y]=weight_data->output[train_pos][y];
            else
                output[y]=0;
        }
        added=1;
        train_pos++;
    }


}

int main ( int argc, char **argv )
{
    srand(time(NULL));
    if ( argc<=1 )
    {
      //  printf ( "neuro num\r\n" );
     //   exit ( 0 );
    }
char filename[255]="train.dat";
    if (argc>1)
    {
			strcpy(filename,argv[1]);
        //desired_error=atof(argv[2]);
        numn=atoi(argv[1]);
        l1n=atoi(argv[2]);
        if (argc>3)
            l2n=atoi(argv[3]);
        if (argc>4)
            l3n=atoi(argv[4]);
        if (argc>5)
            l4n=atoi(argv[5]);
        if (argc>6)
            l5n=atoi(argv[6]);
        if (argc>7)
            l6n=atoi(argv[7]);
    }

    signal ( 2, sig_term );

    srand ( time ( NULL ) );
		
		printf("loading [%s] ",filename);

		
    train_data = fann_read_train_from_file ( filename);
		//printf("[test] ");
  //  test_data = fann_read_train_from_file ( "test.dat" );

    //weight_data=fann_merge_train_data(train_data,test_data);


    int num=0;
    int y;
    int u;
    int x;
    unsigned reject;


    int best_neur;
    //best_neur=fann_length_train_data(weight_data)/weight_data->num_input/weight_data->num_output-numn;
    best_neur=(train_data->num_input+train_data->num_output)/2;

  
    printf ( "\r\ninput: %d, output: %d, neurons: %d bestneur: %d",
            train_data->num_input, train_data->num_output, num_neurons_hidden,
             best_neur);


    classmin=fann_length_train_data(train_data);
    printf("\r\nmap [%d]: \r\n",classmin);
		int classes[10];
		for(i=0;i<10;i++)
			classes[i]=0;
			
    for ( y=0;y<fann_length_train_data(train_data);y++)
    {

				char chars[]={'B','s','.'};
        num=0;
        for (u=0;u<train_data->num_output;u++)
            if (train_data->output[y][u]==1.0f)
						{
							classes[u]++;
							printf("%c",chars[u]);
                num++;
						}
       // if (num<classmin)
         //   classmin=num;

        //printf(" %d=%d ", y, num);
    }
		printf("\r\nclasses [ ");
		for(i=0;i<train_data->num_output;i++)
			printf("%d=%d ",i,classes[i]);
		printf("] ");
	
	/* for(i=0;i<10;i++)
	classes[i]=0;
	
	classmin=fann_length_train_data(test_data);
	printf("\r\ntest map [%d]: \r\n",classmin);
	for ( y=0;y<fann_length_train_data(test_data);y++)
	{
		
		char chars[]={'B','s','.'};
		num=0;
		for (u=0;u<test_data->num_output;u++)
		if (test_data->output[y][u]==1.0f)
		{
			classes[u]++;
			printf("%c",chars[u]);
			num++;
		}
		// if (num<classmin)
		//   classmin=num;
		
		//printf(" %d=%d ", y, num);
	}
	
	printf("\r\nclasses [ ");
	for(i=0;i<train_data->num_output;i++)
	printf("%d=%d ",i,classes[i]);
	printf("] "); */
  
    fann_destroy_train ( train_data );
  // fann_destroy_train ( test_data );
 //   fann_destroy ( ann );

    return 0;
}
