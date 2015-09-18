/*                                .. -._     ._.--.
                       _.. - '   ,   `.- '  '
               .-- ~- '    (        )     .
              (,--   --._.'-. .--._      (   .-
             .n__.-:_,---.         `--._.'`-'
___________ /_oo=oO_  o-c'_______________________________________a:f___ */

//#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <memory.h>
#include <time.h>

#include <fann/doublefann.h>

#define max(a,b) ((a>b) ? a : b)
#define min(a,b) ((a<b) ? a : b)
int training_started=0;
struct fann *trann=NULL;
struct fann_train_data *weight_data,*cln_test_data,*cln_weight_data,*cln_train_data;
struct fann *ann;
struct fann_train_data *train_data, *test_data;
double epoch_mse=0,train_mse=0,test_mse=0;
double desired_error=0.0001f;
double num_train_restarts=1;
int bit_fail_test,bit_fail_train;
double starttemp=20000.0f;
		int param="\x00";
 unsigned log_epochs=0;
double sarprop_temp_factor=0.001f;
double endtemp=0.0000f;
double midtemp;
double sarprop_temp=0;
int each_epoch=0,each_epochs=0;
double test_perc,train_perc;
int effects=0;
double jitter_value;
int jitter_train=1;
int rprop_rand=0;
double jit_factor=0.0001f; //-0.001475
char histfile[]="train_hist.dat";
int num_loop=0;
double diff_mse;
unsigned  tp,prev_tp,min_test_mse_reached=0,last_testsave_epoch=0;
unsigned prev_test_mse2_epoch=0;
int sarprop_rand=0;
double prev_sar_mse=1;
struct fann *sarnet=NULL;
int nextalgo=0;
unsigned min_test_mse_epoch=0;
double good_network_mse=44,prev_test_mse;
unsigned sar_start_epoch=0;
unsigned epochs=0, last_save_epoch=0,save_epoch=40;
double min_mse=1,prev_epoch_mse,prev_mse;
unsigned min_mse_epoch=0,prev_bit_fail_train=0,min_bit_fail_train=0,need_bit_fail=0;
unsigned last_stat_epoch=0;
struct fann *sar_net=NULL,*good_network=NULL;
int min_mse_reached=0;
double test_thr_mse=0;
double min_test_mse=1;
double start_jit_perc=0;
struct fann *init_network;
unsigned non_changed_mse=0;
int auto_train=0;
double src_jittered_val=0;
double maxperc;
char jit_sar=1;
int jit_data=0;
int random_jit=0;
double max_test_perc=1;
unsigned prevsecs=0;
double last_test_mse=0;
int errthan=0;
///////////////////////////////////////////////////////////////////////////
void sig_term ( int p );
double jit_value(double src_val);
int ftest_data(void);
void plot(double p1, double p2,double p3,double p4,double p5,double p6);
void apply_jjit(struct fann_train_data *data, struct fann_train_data *clean_data);
///////////////////////////////////////////////////////////////////////////

int main(int argc,char *argv[])
{
    
    unsigned train_algo=FANN_TRAIN_RPROP;
    int j;
    int reject_total=0;
    int reject=0,y=0,num=0,u=0,x=0,best_neur=0,num_neurons_hidden=0;
    int sar_props=3;

    srand(time(NULL));

    fann_seed_rand();

    //SetConsoleCtrlHandler(ctrlhandler, true);
    signal ( 2, sig_term );

    train_data = fann_read_train_from_file ( "train.dat" );
    test_data = fann_read_train_from_file ( "test.dat" );
     weight_data=fann_merge_train_data(train_data,test_data);
    //cln_weight_data=fann_duplicate_train_data(weight_data);
    cln_test_data=fann_duplicate_train_data(test_data);
    cln_train_data=fann_duplicate_train_data(train_data);
		

    while ((param=getopt ( argc, argv, "sSRrj:qJeQawv:bzt:im")) !=-1)
    {
       // printf( "[%c] \r\n",param);
        switch ((int)param)
        {
					case 'm':
					printf("using thangens error func\r\n");
					errthan=1;
					break;
        case 'z':
            //unlink("active.net");
            ann = fann_create_from_file ( "active.net" );
            fann_clear_train_arrays(ann);
						fann_save(ann,"active.net");
            printf("ok, zeroed active");
						
            exit(0);
            break;
        case 'w':
            // use wedrow&scruchen technique
            fann_init_weights(ann, train_data);
            printf("weight initialized using widrow and NGues mehanism");
            break;
        case 'h':
            printf("helo");
            exit(0);
            break;
        case 'j':
            jit_data=1;
            start_jit_perc=atol(optarg);
            if (start_jit_perc)
                jit_sar=1;
            printf("[contaminating data reactor with <%6.2f%%> noise]\r\n", start_jit_perc);
            break;

        case 'S':
				   sarprop_rand=1;
					train_algo=FANN_TRAIN_SARPROP;
					sarprop_temp=	starttemp;
					endtemp=0.0001f;//starttemp*0.5f;
					sar_props=0;
					 break;
        case 's':

            train_algo=FANN_TRAIN_SARPROP;
            sarprop_temp=	starttemp;

            //	printf("\r\nsarprop init temp %f",fann_get_sarprop_temperature(ann));
            endtemp=0.0001f;//starttemp*0.5f;
            sar_props=0;
           // if (param==0x53)
             

            break;
				case 'e':
				effects=1;
				break;
        case 't':
            sarprop_temp=	atol(optarg);
            starttemp=sarprop_temp;
            break;
        case 'r':
            //start_jit_perc=atoi(optarg);
            train_algo=FANN_TRAIN_RPROP;
            break;

        case 'R':
            train_algo=FANN_TRAIN_RPROP;
            rprop_rand=1;
            //start_jit_perc=atoi(optarg);
            break;

        case 'i':
            train_algo=FANN_TRAIN_INCREMENTAL;
            break;

        case 'q':
            train_algo=FANN_TRAIN_QUICKPROP;
            break;

        case 'Q':
            train_algo=FANN_TRAIN_QUICKPROP;

            break;

        case 'b':
            train_algo=FANN_TRAIN_BATCH;
            break;

        case 'a':
            train_algo=FANN_TRAIN_QUICKPROP;
            auto_train=1;
            break;

        case 'J':
            random_jit=1;
            break;

        case 'v':
            each_epochs=atoi(optarg);
            each_epoch=1;
						log_epochs=each_epochs;
            break;

        default:
            printf("unknown opt %c\r\n",param);
            break;
        }
    }

    if ( ( ann = fann_create_from_file ( "active.net" ) ) !=NULL )
    {
        printf ( "loaded normal network %p. nettype: %u conn_rate: %.2f layers: %u connections: %u neur: %u",ann ,fann_get_network_type(ann),
                 fann_get_connection_rate(ann),
                 fann_get_num_layers(ann),
                 fann_get_total_connections(ann),
                 fann_get_total_neurons(ann));
        //fann_print_connections(ann);

    }
    else
    {

        ann = fann_create_from_file ( "train.net" );
        printf("training new network %p",ann);
        //return 1;
    }

    if (argc>1)
    {

        printf("\r\ntrain: %s [%u]",FANN_TRAIN_NAMES[train_algo],train_algo);

        fann_set_training_algorithm ( ann, train_algo );
        fann_set_sarprop_temperature(ann, starttemp);
    }



    printf("\r\ntrain classes [ ");
    for (y=0;y<train_data->num_output;y++)
    {

        num=0;
        for (u=0;u<fann_length_train_data(train_data);u++)
            if (train_data->output[u][y]==1.0f)
                num++;

        printf(" %d=%d ", y, num);
    }

    for (j=0;j<fann_length_train_data(train_data);j++)
    {
        reject=0;
        for (x=0;x<train_data->num_output;x++)
            if (train_data->output[j][x]<=0.0f)
                reject++;
        //	else if(reject)
        //	printf("no %d ok - %f\n",reject,train_data->output[j][x]);
        if (reject>=train_data->num_output)
        {
            //	printf(" rule %d: %.4f %.4f %.4f\n", j, train_data->output[j][0],
            //	train_data->output[j][1],
            //	train_data->output[j][2]);
            reject_total++;
        }
    }
    printf(" reject=%d", reject_total);
    printf(" ]");

    printf ( "\r\ntrain_data=%p, count: %d input: %d, output: %d, neurons: %d bestneur: %d",
             train_data, fann_length_train_data(train_data), train_data->num_input, train_data->num_output, num_neurons_hidden,
             best_neur);

     if (fann_set_scaling_params(ann, train_data,-1.0f,1.0f,-1.0f, 1.0f)==-1)
       printf("set scaling error\n");
  //  fann_scale_train(ann,train_data);
	if (fann_set_scaling_params(ann, weight_data,-1.0f,1.0f,-1.0f, 1.0f)==-1)
	printf("set scaling error\n");
  //  fann_scale_train(ann,weight_data);
	if (fann_set_scaling_params(ann, test_data,-1.0f,1.0f,-1.0f, 1.0f)==-1)
	printf("set scaling error\n");
		// fann_scale_train(ann,test_data);
    //    if (fann_set_scaling_params(ann, test_data,-1.0f,1.0f,-1.0f, 1.0f)==-1)
    //	printf("set scaling error\n");
   

    /* 	fann_scale_output_train_data(train_data,0.0f,1.0f);
    fann_scale_input_train_data(train_data, -200.0,200.0f);
    	fann_scale_output_train_data(test_data,0.0f,1.0f);
    	fann_scale_input_train_data(test_data, -200.0,200.0f); */
			if(errthan)
				fann_set_train_error_function ( ann, FANN_ERRORFUNC_TANH);
			else
				fann_set_train_error_function ( ann, FANN_ERRORFUNC_LINEAR);
  //  fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
		fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
		fann_set_bit_fail_limit(ann, 0.01f);
    //fann_set_bit_fail_limit ( ann, ( fann_type ) 0.01f );
    //fann_set_callback ( ann, train_callback );
		
		char histfile[]="train_hist.dat";
		FILE *f;
		f=fopen(histfile,"w");
		fputs ("0 0 0 0 0 0 0 0\r\n",f);
		fclose(f);

    int l=1,numn;
    numn=fann_get_num_layers(ann);
    printf("\r\nact funcs: ");
    for (;l<numn;l++)
    {
        int nfunc;
        nfunc=rand()%(sizeof(enum fann_activationfunc_enum )*4);
        if (nfunc==1||nfunc==2)
            nfunc=FANN_LINEAR_PIECE_SYMMETRIC;

        double stp;

        stp=rand()  % 100;

        //if(l==1)
        //	nfunc=FANN_SIGMOID_STEPWISE;

        //	fann_set_activation_steepness_layer(ann, 	0.1+(stp*0.01), l);
//		fann_set_activation_function_layer(ann,nfunc,l);
        printf("<lay#%-02d %s:%-4.02f> ", l, FANN_ACTIVATIONFUNC_NAMES[	fann_get_activation_function(ann,l,0)],
               fann_get_activation_steepness(ann,l,0));

    }
    //printf("");

    need_bit_fail= fann_length_train_data(train_data)/ train_data->num_output;
    min_mse = fann_test_data ( ann, train_data );
    min_bit_fail_train = fann_get_bit_fail ( ann );
    train_mse=min_mse;
    test_mse = fann_test_data(ann, test_data);
    prev_test_mse=test_mse;
    min_test_mse=3;
    prev_bit_fail_train=min_bit_fail_train;
    // fann_reset_MSE ( ann );
    printf("\r\nstart training [minmse: %.6f (%.6f) minbf: %u needbf: %u testmse: %.8f]\r\n",
           min_mse,min_mse*0.9999f,min_bit_fail_train,need_bit_fail,test_mse);
    min_mse=min_mse*0.9991f;
    double rdec,rinc;
    good_network=fann_copy(ann);
    //	good_network=NULL;
    static double prev_test_epoch_mse;
    //prev_test_mse2=test_mse;
    good_network_mse=min_mse;
    sar_start_epoch=500;
    unlink(histfile);
    sarnet=fann_copy(ann);
    // fann_set_learning_rate ( ann, 0.1f);
    //  fann_set_learning_momentum(ann, 0.4f);

    init_network=fann_copy(ann);
    while (true)
    {

        epoch_mse = fann_train_epoch(ann, train_data);

        if (epoch_mse!=prev_test_epoch_mse)
        {
            prev_test_epoch_mse=epoch_mse;
            num_loop=0;
        }
        else if (num_loop++>1115)
        {
            printf("\r\n [restored train %d] - loop detected\r\n ",train_algo);

            fann_destroy(ann);
            ann = fann_copy(sarnet);


            num_loop=0;
            nextalgo=1+(rand()%4);
            fann_set_training_algorithm(ann,train_algo);
        }
        epochs++;
        ftest_data();

        if (epoch_mse<=desired_error)
        {
            printf("\r\nreached target mse %.8f",epoch_mse);

        }

        diff_mse=100.0f-((min(train_mse,test_mse)/max(test_mse,train_mse))*100.0f);

        if (epochs-last_save_epoch>25)
        {
            //printf(" save train net ",ann);
            fann_save ( ann, "train.net" );
            last_save_epoch=epochs;


        }

        if (epoch_mse<min_mse)
        {
            test_mse = fann_test_data(ann, test_data);
            min_mse_reached++;

            min_mse_epoch=epochs;



            min_mse=epoch_mse;
            if (non_changed_mse)
                non_changed_mse--;
        }
        else
        {
            static unsigned ncmtime=0;
            //if(time(NULL)-ncmtime>=1)
            {
                ncmtime=time(NULL);
                if (non_changed_mse<100)
                    non_changed_mse++;
            }
        }


        if (test_mse<min_test_mse)
        {
            //	test_mse = fann_test_data(ann, test_data);
            min_test_mse_reached++;

            min_test_mse_epoch=epochs;
            if (epochs)
                min_test_mse=test_mse;
            if (good_network_mse>test_mse)
            {
                if (good_network)
                    fann_destroy(good_network);
                //	ann=good_network;
                good_network=fann_copy(ann);
                //	good_network=NULL;
                good_network_mse=test_mse;
                //			printf("[save <%f0.8f@%p>] ",good_network_mse,good_network);
            }
            if (trann)
                fann_destroy(trann);
            trann=fann_copy(ann);

            if (epochs-last_testsave_epoch>5)
            {

                fann_save ( ann, "test.net" );
                last_testsave_epoch=epochs;


            }

            //	min_mse=epoch_mse;
        }


            if (fann_get_training_algorithm(ann)==FANN_TRAIN_SARPROP)//||sar_props>1))bit_fail_train>=prev_bit_fail_train||
            {

                
                if (sarprop_rand&&rand()%3==2)
                {
                    fann_set_sarprop_weight_decay_shift(ann,((rand()%100)* ((rand()%2) ? 0.1f:-0.1f)));
                    fann_set_sarprop_step_error_threshold_factor(ann,(rand()%100)*0.01f);
                    fann_set_sarprop_step_error_shift(ann, ((rand()%100)*0.01f));
                }
                // fann_set_sarprop_temperature(ann,temp);


            }
     
			sarprop_temp-=sarprop_temp_factor;
			
			fann_set_sarprop_temperature(ann, sarprop_temp);

        if (test_perc>max_test_perc)
        {
            max_test_perc=test_perc;
            fann_save(ann,"best-test.net");
        //    printf("S");
        }
static int numdoneepochs=0;

        if (time(NULL)-prevsecs>=2||(numdoneepochs++>log_epochs))
        {
					numdoneepochs=0;
            each_epochs=each_epoch;
            unsigned uu,yy;
						yy=(1+(unsigned)rand())%((unsigned)rand()+1);
            for (unsigned xx=0;xx<yy;xx++)
                uu=rand();
            argv[0]=uu;


            prevsecs=time(NULL);
            printf("\r\n");
            if (epoch_mse!=prev_mse)
            {

                //	prev_mse=epoch_mse;
            }

           /*  if (fann_get_training_algorithm(ann)==FANN_TRAIN_SARPROP&&(sarprop_temp<=endtemp)||epoch_mse==0.5f)
            {

                printf(" [sar cycle] ");
                //	if(fann_get_sarprop_temperature(ann)<=0.0f)
                if (fann_get_training_algorithm(ann)==FANN_TRAIN_SARPROP)
                {

                    if (!min_mse_reached)
                    {
                        sarprop_temp_factor=(rand()%100)*0.001f;
                        starttemp=rand()%2000;
                        endtemp=1.0f;//starttemp*0.5f;
                        if (epoch_mse==0.5f||epoch_mse==0.62496680f)
                        {
                            ann=fann_copy(init_network);

                            fann_set_training_algorithm(ann,FANN_TRAIN_SARPROP);
                        }

                    }
                    //		} else {
                    //		fann_set_sarprop_temperature(ann,starttemp);
                }

                sarprop_temp=starttemp;
                fann_set_sarprop_temperature(ann,starttemp);
                if (min_mse_reached)
                {
                    printf(" save train ");
                    fann_save(ann,"train.net");
                }

            } */



            static unsigned lasttrainingepoch=0;

            prev_tp=epochs;

            int nfunc=0;
            double train_thr_mse=0;
            fann_reset_MSE(ann);
            train_mse=fann_test_data(ann,train_data);

            fann_reset_MSE(ann);
            test_mse=fann_test_data(ann,test_data);

            //    nfunc=fann_get_activation_function(ann, 2, 0);
            int stpns;
            //	stpns=fann_get_activation_steepness(ann,1,0);
            //	printf("\r\n%f",diff_mse*0.1f);
            //	fann_set_activation_steepness_layer(ann, diff_mse*0.1f, 1);
            //   fann_set_activation_function_layer(ann,FANN_THRESHOLD_SYMMETRIC,2);
            //   fann_reset_MSE(ann);
            //   test_thr_mse=fann_test_data(ann,test_data);
            //    fann_reset_MSE(ann);
            //    train_thr_mse=fann_test_data(ann,train_data);
            //fann_set_activation_steepness_layer(ann, 	stpns, 1);
            // fann_set_activation_function_layer(ann,nfunc,2);

            if (test_mse<min_test_mse)
            {
                //min_test_mse=test_mse;
                min_test_mse_reached++;
            }
            if (min_mse_reached)
                printf(" minmse: %.8f reached: %3u ",min_mse,epochs-min_mse_epoch);
            min_mse_reached=0;
            if (min_test_mse_reached)
                printf(" mintestmse: %.8f reached: %3u ",min_test_mse,min_test_mse_reached);
            min_test_mse_reached=0;

            int test_course=0;
            //
            bit_fail_train = fann_get_bit_fail ( ann );

            char updown;
            if (prev_mse>epoch_mse&&prev_test_mse>test_mse)
                updown='x';
            else if (prev_mse>epoch_mse&&prev_test_mse<test_mse)
                updown='~';
            else if (prev_mse<epoch_mse)
                updown='-';
            printf("\r%6d %-10.8f (%.8f) %6.2f%% |%c| %-10.8f (%.8f) %6.2f%% ",
                   epochs,epoch_mse,  min_mse,train_perc,updown,
                   test_mse,min_test_mse,test_perc);

            if (fann_get_training_algorithm(ann)==FANN_TRAIN_SARPROP)
            {
                printf("sarwd: %8.4f sareth: %5.3f sares: %5.4f sartemp: %9.6f ", fann_get_sarprop_weight_decay_shift(ann),
                       fann_get_sarprop_step_error_threshold_factor(ann),
                       fann_get_sarprop_step_error_shift(ann),
                       fann_get_sarprop_temperature(ann));
            }

            //	fann_set_activation_steepness_layer(ann, (1.0f-train_mse), 2);

            if ((!random_jit&&jit_data&&epochs>2&&start_jit_perc) || (random_jit && rand()%7==3))
            {
                apply_jjit(train_data, cln_train_data);
                printf("[jit %.8f => %.8f]",src_jittered_val,jitter_value);
                //} else {
                //fann_destroy_train(train_data);
                //	train_data=fann_duplicate_train_data(cln_train_data);
            }



            //		rinc+=diff_mse*0.000001f;
            if (rprop_rand&&rand()%2==1&&epochs>1)
            {

                rdec=0.0001f+((rand()%5000)*0.0001f);
                if (!rdec)
                    rdec=0.01f;
                rinc=0.0001f+((rand()%190)*0.01f);
                if (!rinc)
                    rinc=1.1f;

                fann_set_rprop_increase_factor(ann,rinc );
                fann_set_rprop_decrease_factor(ann, rdec);
                fann_set_learning_rate ( ann, 0.01f+((rand()%98)*0.01f));//0.9f );
                fann_set_learning_momentum(ann, 0.01f+((rand()%98)*0.01f));

                printf("[rprop inc %.4f dec %.4f lrn %.2f mom %.2f]", fann_get_rprop_increase_factor(ann),
                       fann_get_rprop_decrease_factor(ann),
                       fann_get_learning_rate(ann),
                       fann_get_learning_momentum(ann));

            }


            fann_save(ann,"active.net");
						//printf("x");
        } else {
					
				//	if(effects)
					//{
					//	if(prev_mse<epoch_mse)
							//printf(".");
					//	else
							
					
				}
        plot(epochs,epoch_mse,test_mse,test_perc/100.0f,train_perc/100.0f,min_mse);

        prev_mse=epoch_mse;
        prev_bit_fail_train=bit_fail_train;

    }

    return 0;
}
void plot(double p1, double p2,double p3,double p4,double p5,double p6)
{
    FILE *f;
    f=fopen(histfile, "a");
    char str[128];
    sprintf(str,"%f %.8f %.8f %.8f %.8f %.8f\n",p1,p2,p3,p4,p5,p6);
    fwrite(str, strlen(str),1,f);
    fclose(f);
}
double jit_value(double src_val)
{
    double jpercint;

    if (!start_jit_perc)
        return src_val;
    static double prevrnd;
    double rnd=1;

    rnd=rand();
    while (rnd==prevrnd||!rnd)
        rnd=(double)rand();
    prevrnd=rnd;

    maxperc=(double) ((unsigned)rnd%(unsigned)start_jit_perc);
    jpercint=(src_val*(double)maxperc)/(double)100.0f;

    src_jittered_val=src_val;
    if (rand()%6==2)
        jitter_value=src_val+jpercint;
    else
        jitter_value=src_val-jpercint;

    //(double) ((double)src_val															 +
    //										 (double)( (rand()%(unsigned long)()*800000000) *0.00000001)																	);
    return jitter_value;
}
void apply_jjit(struct fann_train_data *data, struct fann_train_data *clean_data)
{
    int i;

    int inc;
    // inc=rand()%2;

//	printf("[jit %f] ",jitt_value);
    //exit(0);
    for (i=0;i<fann_length_train_data(clean_data);i++)
    {
        int x;

        for (x=0;x<data->num_input;x++)
        {
            data->input[i][x]=jit_value(clean_data->input[i][x]);//-jitter_value;

        }
    }
}
int ftest_data(void)
{    //	sar_start_epoch=0;
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
            //    calc_out2[i]*=2.0f;
            if (calc_out2[i]<minv)
            {
                minv=calc_out2[i];
                minat=i;
            }
            if (calc_out2[i]>maxv)
            {
                maxv=calc_out2[i];
                maxat=i;
            }
        }

        int ok=0;
        ok=0;
        for (i=0;i<train_data->num_output;i++)
            if (train_data->output[calc2][i]>0.0&&maxat==i)
                ok=1;

        if (ok)success++;
        else
            fails++;

    }
    train_perc=(success/fann_length_train_data(train_data))*100.0f;
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
            //    calc_out2[i]*=2.0f;

            if (calc_out2[i]<minv)
            {
                minv=2*calc_out2[i];
                minat=i;
            }
            if (calc_out2[i]>maxv)
            {
                maxv=calc_out2[i];
                maxat=i;
            }
        }

        int ok=0;
        ok=0;
        for (i=0;i<test_data->num_output;i++)
            if (test_data->output[calc2][i]>0.0&&maxat==i)
                ok=1;

        if (ok)success++;
        else
            fails++;

    }
    test_perc=(success/fann_length_train_data(test_data))*100.0f;
    //printf(" fails %.0f success %.0f (%5.2f%%) ",
    //fails,success,perc
    //);

    // fann_set_activation_function_hidden ( ann,  rand()*0.81);
    // printf("\r\n rpropfact dec/inc r %.5f %.5f lr %.5f mom %.5f",fann_get_rprop_decrease_factor(ann),fann_get_rprop_increase_factor(ann), fann_get_learning_rate ( ann),
    //       fann_get_learning_momentum(ann));

    //	rebuild_functions();

}

void sig_term ( int p )
{    //  printf ( "\r\nsaving net...\r\n" );
    //  fann_save ( ann, "train.net" );

    //SleepEx(1000*1000*4, false);
    fann_destroy(ann);
    fann_destroy_train(test_data);
    fann_destroy_train(weight_data);
    fann_destroy_train(train_data);
//   ExitProcess(0);
}