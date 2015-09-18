#include <stdio.h>
#include <signal.h>
#include <time.h>
#include <memory.h>
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
void rebuild_functions(int neur);
unsigned train_pos = 0;
unsigned finaltestdatanum=0;
unsigned *train_matrix;
double test_perc,train_perc;


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

int ftest_data(void)
{
    //	sar_start_epoch=0;
    //  printf("\r\n\r\n--------------------------------------------------------------------------------");

    double val_2[10];
    fann_type *calc_out2;
    unsigned calc2;
    int curi=0;
    double fails=0,success=0;
    double perc=0;
    double minv=9,maxv=-1;
    int i;
    int minat=0,maxat=0;

    for (curi=0;curi<fann_length_train_data(train_data);curi++)
    {

        calc2=curi;//rand()%(fann_length_train_data(train_data)-1);
        //printf("\r\ntesting %u %u ",calc1,calc2);
        //fann_scale_input(ann, test_data->input[calc1]);
        //fann_scale_input(ann, train_data->input[calc2]);
        //	fann_scale_output(ann, test_data->input[calc1]);

        //fann_scale_input(ann, train_data->input[calc2]);
        calc_out2 = fann_run(ann, train_data->input[calc2]);
        //	fann_descale_output(ann,calc_out2);

        memcpy(&val_2,  calc_out2, sizeof(double)*3);





        minv=9;
        maxv=-1;
        for (i=0;i<train_data->num_output;i++)
        {
            if (val_2[i]<minv)
            {
                minv=val_2[i];
                minat=i;
            }
            if (val_2[i]>maxv)
            {
                maxv=val_2[i];
                maxat=i;
            }
        }

        int ok=0;
        ok=0;
        for (i=0;i<train_data->num_output;i++)
            if (train_data->output[calc2][i]==1&&maxat==i)
                ok=1;

        if (ok)success++;
        else
            fails++;

    }
    train_perc=((double)success/(double)fann_length_train_data(train_data))*100.0f;
    //printf(" fails %.0f success %.0f (%5.2f%%) ",
    //fails,success,perc
    //);

    fails=0;
    success=0;

    for (curi=0;curi<fann_length_train_data(test_data);curi++)
    {

        calc2=curi;//rand()%(fann_length_train_data(train_data)-1);
        //printf("\r\ntesting %u %u ",calc1,calc2);
        //fann_scale_input(ann, test_data->input[calc1]);
        //fann_scale_input(ann, train_data->input[calc2]);
        //	fann_scale_output(ann, test_data->input[calc1]);

        //fann_scale_input(ann, train_data->input[calc2]);
        calc_out2 = fann_run(ann, test_data->input[calc2]);
        //	fann_descale_output(ann,calc_out2);

        memcpy(&val_2,  calc_out2, sizeof(double)*3);





        minv=9;
        maxv=-1;
        for (i=0;i<test_data->num_output;i++)
        {
            if (val_2[i]<minv)
            {
                minv=val_2[i];
                minat=i;
            }
            if (val_2[i]>maxv)
            {
                maxv=val_2[i];
                maxat=i;
            }
        }

        int ok=0;
        ok=0;
        for (i=0;i<test_data->num_output;i++)
            if (test_data->output[calc2][i]==1&&maxat==i)
                ok=1;

        if (ok)success++;
        else
            fails++;

    }
    test_perc=((double)success/(double)fann_length_train_data(test_data))*100.0f;
    //printf(" fails %.0f success %.0f (%5.2f%%) ",
    //fails,success,perc
    //);

    // fann_set_activation_function_hidden ( ann,  rand()*0.81);
    // printf("\r\n rpropfact dec/inc r %.5f %.5f lr %.5f mom %.5f",fann_get_rprop_decrease_factor(ann),fann_get_rprop_increase_factor(ann), fann_get_learning_rate ( ann),
    //       fann_get_learning_momentum(ann));

    //	rebuild_functions();


}
int main ( int argc, char **argv )
{
    srand(time(NULL));
    if ( argc<=1 )
    {
        //      printf ( "neuro num\r\n" );
        //     exit ( 0 );
    }

    if (argc>2)
    {
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

    printf("loading training data...");

    train_data = fann_read_train_from_file ( "train.dat" );
    test_data = fann_read_train_from_file ( "test.dat" );

    weight_data=fann_merge_train_data(train_data,test_data);

    cln_weight_data=fann_duplicate_train_data(weight_data);
    cln_test_data=fann_duplicate_train_data(test_data);
    cln_train_data=fann_duplicate_train_data(train_data);

    //num_neurons_hidden = atoi ( argv[1] );

    unsigned epochs=0;
    double best_perc;
    double prev_epoch_mse=1;
    double got_inc;
    int num=0;
    int y;
    int u;
    int x;
    unsigned reject;
    double train_mse;
    unsigned last_stat_epoch=0;
    srand(time(NULL));
    struct fann *good_ann;
    while (true)

    {
        for (y=1;y<215;y++)
        {
            conn_rate=0.5f+((rand()%50)*0.01f);
            //  printf("create network: layers=%d l1n=%d l2n=%d l3n=%d l4n=%d\ l5n=%d l6n=%dr\n",numn,l1n,l2n,l3n,l4n,l5n,l6n);
            ann = fann_create_standard (//conn_rate,
                      3,
                      train_data->num_input,
                      y,
                      train_data->num_output );
            fann_init_weights ( ann, train_data );

            if ( ( int ) ann==NULL )
            {
                printf ( "error" );
                exit ( 0 );
            }
            printf("\r\n%4d: ",y);

            fann_set_activation_function_hidden(ann,FANN_SIGMOID);
            fann_set_activation_function_output(ann,FANN_SIGMOID);

            //  rebuild_functions(y);
            fann_set_training_algorithm ( ann, FANN_TRAIN_RPROP );

            got_inc=0;
            prev_epoch_mse=1;
            best_perc=1;
            epochs=0;

            good_ann=fann_copy(ann);
            for (u=0;u<100;u++)
            {
                train_mse=fann_train_epoch(ann, train_data);
                ftest_data();

                if (test_perc>best_perc)
                {
                    //	u--;


                    //		printf("%f",test_perc);
                    //	fann_destroy(ann);
                    //	ann=fann_copy(good_ann);
                    best_perc=test_perc;
                    printf("%6.2f",test_perc);

                }
                else                {
                    //best_perc=test_perc;
                    //		printf("x");
                    printf(".");

                    if (epochs-last_stat_epoch>=50)
                    {

                        last_stat_epoch=epochs;
                    }
                    //		fann_destroy(ann);
                    //	ann=fann_copy(good_ann);
                    //fann_randomize_weights ( ann, -((rand()%100)*0.01f), ((rand()%100)*0.01f) );
                }
                got_inc=test_perc-prev_epoch_mse;
                prev_epoch_mse=best_perc;

                epochs++;
            }
            printf(" %6.2f inc: %.2f",test_perc,got_inc);
//            printf("%6.2f %6.2f",train_perc,test_perc);

            fann_destroy ( ann );
        }
    }

    fann_destroy_train ( train_data );
    fann_destroy_train ( test_data );
    fann_destroy ( ann );

    return 0;
}

void rebuild_functions(int neur)
{
    int sygm_functions[]={FANN_SIGMOID_SYMMETRIC_STEPWISE,FANN_SIGMOID_SYMMETRIC};
    int sym_functions[]={FANN_LINEAR,
                         FANN_GAUSSIAN_SYMMETRIC,FANN_COS_SYMMETRIC,FANN_SIN_SYMMETRIC,
                         FANN_LINEAR_PIECE_SYMMETRIC,FANN_ELLIOT_SYMMETRIC,
                         FANN_SIGMOID_SYMMETRIC_STEPWISE,FANN_SIGMOID_SYMMETRIC
                        };

    int functions[]={FANN_ELLIOT,FANN_LINEAR,FANN_GAUSSIAN,FANN_COS,FANN_SIN,
                     FANN_SIGMOID_STEPWISE,FANN_LINEAR_PIECE,FANN_SIGMOID,FANN_GAUSSIAN_STEPWISE,

                     FANN_LINEAR,
                     FANN_GAUSSIAN_SYMMETRIC,FANN_COS_SYMMETRIC,FANN_SIN_SYMMETRIC,
                     FANN_LINEAR_PIECE_SYMMETRIC,FANN_ELLIOT_SYMMETRIC,
                     FANN_SIGMOID_SYMMETRIC_STEPWISE
                    };

    int mid_functions[]={FANN_SIGMOID_STEPWISE,FANN_ELLIOT,FANN_LINEAR_PIECE,
                         FANN_GAUSSIAN_STEPWISE,FANN_GAUSSIAN,FANN_COS,FANN_SIN,FANN_SIGMOID
                        };

    int in_functions[]={FANN_SIGMOID_SYMMETRIC,FANN_SIGMOID_SYMMETRIC_STEPWISE,FANN_GAUSSIAN_SYMMETRIC};
    int out_functions[]={FANN_GAUSSIAN_SYMMETRIC,FANN_SIGMOID_SYMMETRIC,FANN_SIGMOID_SYMMETRIC_STEPWISE};

    int l=1,a=0;



    numn=fann_get_num_layers(ann);
    // printf("\r\n[ act funcs: ");
    for (l=1;l<2;l++)
    {
        int sta;

        if (l==1)
            sta=neur;
        else
            sta=2;
        for (a=0;a<sta;a++)
        {
            int nfunc;
            if (l==1)
                nfunc=functions[rand()%((sizeof(functions)/sizeof(int)))];
            else if (l==numn-1)
                nfunc=functions[rand()%((sizeof(functions)/sizeof(int)))];
            else
                nfunc=functions[rand()%((sizeof(functions)/sizeof(int)))];
            //	printf("nfunc %d",nfunc);
            //   if (nfunc==1||nfunc==2)
            //     nfunc=FANN_LINEAR_PIECE_SYMMETRIC;

            double stp;

            stp=rand()  % 100;

            //if(l==1)
            //	nfunc=FANN_SIGMOID_STEPWISE;
            stp=0.1+(stp*0.01);
            if (l==numn-1||l==1)
                stp=1.0f;
            else
                stp=1.0f;
            fann_set_activation_steepness_layer(ann, 	stp, l);

            char chars[]={'q','w','e','r','t','y','u','i','o','z','x','c','v','b','n','a','s','d','f','g','h'};

            printf("%c",chars[nfunc]);
            //	printf("\r\nset %d %d",l,a);
            fann_set_activation_function(ann,nfunc,l,a);
            //   printf("\r\n #%-02d %s <%-4.02f55l, FANN_ACTIVATIONFUNC_NAMES[	fann_get_activation_function(ann,l,0)],
            //       fann_get_activation_steepness(ann,l,0));

        }
    }
    // printf("]\r\n");



}