#include <stdio.h>
#include <signal.h>
#include <time.h>
#include <memory.h>
#include <fann/doublefann.h>
#include <unistd.h>

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
double weight_mse,test_mse;
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
char histfile[]="mutate_hist.dat";
double jitt_value;
unsigned epochs=0;
double best_perc;
double prev_epoch_mse=1;
double got_inc;
int jitter_train=1;
int y;
double jitter_value=0.0001f;
int x;
unsigned reject;
double train_mse;
unsigned last_stat_epoch=0;
unsigned lay,ln=0,y2=0,neur1,neur2;
struct fann *good_ann=NULL;
void plot(double p1, double p2,double p3)
{
    FILE *f;
    f=fopen(histfile, "a");
    char str[128];
    sprintf(str,"%f %.8f %.8f\n",p1,p2,p3);
    fwrite(str, strlen(str),1,f);
    fclose(f);
}

void apply_jjit(struct fann_train_data *data, struct fann_train_data *clean_data)
{
    int i;




//	printf("[jit %f] ",jitt_value);
    //exit(0);
    for (i=0;i<fann_length_train_data(clean_data);i++)
    {
        int x;

        for (x=0;x<data->num_input;x++)
        {
            if (rand()%3)jitt_value=((rand()%10000)*jitter_value);
            if (rand()%2)
                data->input[i][x]=clean_data->input[i][x]-jitt_value;
            else
                data->input[i][x]=clean_data->input[i][x]+jitt_value;
        }
    }
}
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
    unsigned fails=0,success=0;
    double perc=0;
    double minv=9,maxv=-1;
    int i;
    int minat=0,maxat=0;

    test_mse=fann_test_data(ann,test_data);

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
            if ((double)calc_out2[i]<minv)
            {
                minv=val_2[i];
                minat=i;
            }
            if ((double)calc_out2[i]>maxv)
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
    /*   printf(" fails %5u success %5u (%5.2f%%) ",
             fails,success,train_perc
            ); */

    fails=0;
    success=0;
    unsigned failed_classes[10];

    for (curi=0;curi<test_data->num_output;curi++)
        failed_classes[curi]=0;

    int nfunc=0;
    double train_thr_mse=0;


    nfunc=fann_get_activation_function(ann, 3, 0);
    int stpns;
    stpns=fann_get_activation_steepness(ann,1,0);
    //	printf("\r\n%f",diff_mse*0.1f);
    //fann_set_activation_steepness_layer(ann, 0.3f, 1);
    //fann_set_activation_function_layer(ann,FANN_THRESHOLD_SYMMETRIC,3);




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
        {
            if (test_data->output[calc2][i]==1&&maxat==i)
                ok=1;
            else if (test_data->output[calc2][i]==1&&maxat!=i)
                failed_classes[i]++;
        }

        if (ok)success++;
        else
            fails++;

    }
    test_perc=((double)success/(double)fann_length_train_data(test_data))*100.0f;
    /*   printf(" fails %5u success %5u (%5.2f%%) [fails: ",
             fails,success,test_perc
            );
      for (curi=0;curi<test_data->num_output;curi++)
          printf("%4u ",failed_classes[curi]);
      printf("] "); */
    // fann_set_activation_function_hidden ( ann,  rand()*0.81);
    // printf("\r\n rpropfact dec/inc r %.5f %.5f lr %.5f mom %.5f",fann_get_rprop_decrease_factor(ann),fann_get_rprop_increase_factor(ann), fann_get_learning_rate ( ann),
    //       fann_get_learning_momentum(ann));

    //	rebuild_functions();

    fann_set_activation_function_layer(ann,nfunc,3);
    fann_set_activation_steepness_layer(ann,stpns, 1);
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



    srand(time(NULL));

    y=atoi(argv[2]);

    lay=atoi(argv[1]);
    ln=lay+2;
    if (lay==1)
        y2=train_data->num_output;
    best_perc=1;

    printf("\r\ndoing %ux%u [layers=%u,out=%u]",lay,y,ln, train_data->num_output);
    while (true)

    {
        neur1=1+(rand()%y);
        neur2=1+(rand()%y);
        conn_rate=0.5f+((rand()%50)*0.01f);
        printf("\r\n%2dx%-4d: ",neur1,neur2);
        //  printf("create network: layers=%d l1n=%d l2n=%d l3n=%d l4n=%d\ l5n=%d l6n=%dr\n",numn,l1n,l2n,l3n,l4n,l5n,l6n);
        ann = fann_create_standard (//conn_rate,
                  ln,
                  train_data->num_input,
                  neur1,
                  neur2,
                  train_data->num_output );
        //fann_init_weights ( ann, train_data );
        printf(" [%p] ",ann);

        if ( ( int ) ann==NULL )
        {
            printf ( "error" );
            exit ( 0 );
        }



        fann_set_activation_function_hidden(ann,FANN_SIGMOID);
        fann_set_activation_function_output(ann,FANN_SIGMOID);

        rebuild_functions(neur1);
        fann_set_training_algorithm ( ann, FANN_TRAIN_RPROP );
        fann_set_sarprop_temperature(ann,15000.0f);
        //fann_randomize_weights ( ann, -((rand()%10)*0.1f), ((rand()%10)*0.1f) );
        fann_init_weights(ann,train_data);
        got_inc=0;
        prev_epoch_mse=1;
        //
        epochs=0;

        unsigned last_best_perc_epoch=0;
        unsigned last_sync_epoch=0;
        unsigned last_ftest_secs=0;

        last_sync_epoch=0;
        last_best_perc_epoch=0;
        if (good_ann)
            fann_destroy(good_ann);

        good_ann=fann_copy(ann);
        unlink(histfile);
        for (u=0;u<1000;u++)
        {
            fflush(NULL);
            train_mse=fann_train_epoch(ann, train_data);

            if (jitter_train)
                apply_jjit(train_data,cln_train_data);


            if (time(NULL)-last_ftest_secs>=1)
            {
                //printf("\r\n%5u %9.6f %5.2f ",epochs,train_mse,test_perc);
                //printf(" %4.2f",test_perc);
                printf(".");


                last_ftest_secs=time(NULL);
            }
            ftest_data();
            plot(epochs,train_mse,test_mse);

            /*         if (epochs>10&&((int)test_perc==43||(int)test_perc==57))
                    {
                        printf(" [excluded %.2f] ",test_perc);
                        break;
                    }            else            {

                    } */
            //printf("excluded %f ",test_perc);

            double prev_test_perc;
            //   if (prev_epoch_mse==best_perc)
            //  printf("o");
            if ((int)test_perc>(int)train_perc&&epochs-last_stat_epoch>10)
            {
                fann_destroy(good_ann);
                good_ann=fann_copy(ann);

                if (test_perc!=prev_test_perc)
                    printf("%.2f [%f]",test_perc,train_mse);

                //printf(" sync[%4.2f]",test_perc);
                last_stat_epoch=epochs;
            }
            else 	if (epochs-last_sync_epoch>111500)
            {
                last_sync_epoch=epochs;
            }
            if (epochs>210&&test_perc>best_perc)
            {
                //	u--;
                //  fann_destroy(good_ann);
                //   good_ann=fann_copy(ann);
                printf(" [saved best %.0f] ",test_perc);
                last_stat_epoch=epochs;
                //		printf("%f",test_perc);
                //	fann_destroy(ann);
                //	ann=fann_copy(good_ann);
                fann_save(ann,"mutate-best.net");
                best_perc=test_perc;
                printf(" %6.2f [%f]",test_perc,train_mse);
                last_best_perc_epoch=epochs;

            }
            else     if (epochs>11100&&((int)test_perc<=63||(int)test_perc==(int)prev_test_perc))
            {
                //best_perc=test_perc;
                //		printf("x");
                //  printf(".");
                //printf("\r%6.8f",train_mse);
                //			printf("done\r\n");
                break;


            }
            static unsigned last_restore_epoch=0;
            if (epochs>100&&test_mse-train_mse>=0.25f&&epochs-last_restore_epoch>=120)
            {
                /* 	fann_set_learning_rate ( ann,0.31f+(rand()%90)*0.01f);
                	fann_set_learning_momentum(ann,(rand()%90)*0.01f);
                	printf(" [restored @ %u lr %.2f mm %.2f]",epochs,fann_get_learning_rate(ann),
                	fann_get_learning_momentum(ann));
                	fann_destroy(ann);
                	ann=fann_copy(good_ann);
                	last_stat_epoch=epochs;
                	last_restore_epoch=epochs; */





                double rdec,rinc;
                rdec=0.0101f+((rand()%100)*0.00001f);
                if (!rdec)
                    rdec=0.01f;
                rinc=1.0001f+((rand()%90)*0.00001f);
                if (!rinc)
                    rinc=1.1f;
                static double prev_test_epoch_mse;

                //		rinc+=diff_mse*0.000001f;
                //			fann_set_rprop_increase_factor(ann,rinc );
                //	fann_set_rprop_decrease_factor(ann, rdec);
            }
            else if (test_mse-train_mse<=0.1f)
            {
                fann_destroy(good_ann);
                good_ann=fann_copy(ann);
                //	printf("s");
            }
            else
            {
                fann_set_sarprop_temperature(ann,fann_get_sarprop_temperature(ann)-0.0001f);
            }
            static unsigned last_train_change_epoch=0;
            if (test_mse>=train_mse&&epochs-last_train_change_epoch>=100)
            {
                last_train_change_epoch=epochs;
                //fann_set_training_algorithm(ann,FANN_TRAIN_SARPROP);
                jitter_train=0;
            }
            else
            {
                //fann_set_training_algorithm(ann,FANN_TRAIN_RPROP);
                jitter_train=0;
            }

            got_inc=test_perc-prev_epoch_mse;
            prev_epoch_mse=test_perc;
            prev_test_perc=test_perc;
            epochs++;
            if (epochs-last_best_perc_epoch>511500)
            {
                printf(" failed");
                break;
            }
            if (epochs>2200&&(int)train_perc<40)
            {
                printf("skip 1\r\n");
                break;
            }

            if ((int)test_perc>=80)
            {
                printf("\r\ngot it %f\r\n",test_perc);
                fann_save(ann,"good.net");
                exit(0);
            }
            // printf("\n%6u ",epochs);
        }
        printf(" %6.2f inc: %.2f",test_perc,got_inc);
//            printf("%6.2f %6.2f",train_perc,test_perc);

        fann_destroy ( ann );

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
                nfunc=mid_functions[rand()%((sizeof(mid_functions)/sizeof(int)))];
            else if (l==numn-1)
                nfunc=mid_functions[rand()%((sizeof(mid_functions)/sizeof(int)))];
            else
                nfunc=mid_functions[rand()%((sizeof(mid_functions)/sizeof(int)))];
            //	printf("mid_functions %d",nfunc);
            //   if (nfunc==1||nfunc==2)
            //     nfunc=FANN_LINEAR_PIECE_SYMMETRIC;

            double stp;

            stp=rand()  % 100;

            //if(l==1)
            //	nfunc=FANN_SIGMOID_STEPWISE;
            stp=0.1+(stp*0.01);
            if (l==numn-1||l==1)
                stp=1.0f;
            // else
            //  stp=1.0f;
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