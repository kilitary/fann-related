#include <stdio.h>
#include <signal.h>
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
unsigned lowest_test_mse_epoch=0;
unsigned nextalgo=0;
unsigned func_num=0;
double stagn_epoch=0;
unsigned prevbitfail=0;
double prevsarep=0,prev_mse_test=0;
double stpns;
unsigned stpns_epoch=0;
double mse_chg=0;
unsigned lastmsechecktime=0;
double minutes_left=0;
double prev_mse_chg[61];
unsigned cur_mse_chg=0;
double last_min_timeleft;
unsigned last_min_timeleft_upd=0;
double weight_mse;
struct fann_train_data *weight_data,*cln_test_data,*cln_weight_data,*cln_train_data;
unsigned l1n=0,l2n=0,l3n=0,l4n=0,l5n=0,l6n=0;
unsigned numn=3;
double conn_rate=1.0f;
unsigned finaldatanum=0;
unsigned reject_total=0;
unsigned num,u;
unsigned classmin=0;
struct fann_train_data * final_data,*final_test_data;
unsigned train_classes_added[10];
void rebuild_functions(void);
unsigned train_pos = 0;
unsigned finaltestdatanum=0;
unsigned *train_matrix;
unsigned mintest=1044440,maxtest=0;


void sig_term ( int p )
{
    printf ( "\r\nsaving net...\r\n" );
    fann_save ( ann, "bb-normal.net" );
    exit ( 0 );
}

void 	train_func( unsigned int num, unsigned int numinp, unsigned int numout, fann_type * input, fann_type * output)
{


    unsigned addthis;
    unsigned i=0;
    unsigned added=0;

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
          //  fprintf(stderr,".");
            train_matrix[train_pos]=1;
            train_pos++;
            finaltestdatanum++;

            continue;
        }

       // fprintf(stderr,"x");
        train_matrix[train_pos]=0;
        unsigned y;
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

    // if (num>finaldatanum)
    //  return;

    unsigned addthis;
    unsigned i=0;
    unsigned added=0;
unsigned y;
    while (!added)
    {
        addthis=1;

        if (!train_matrix[train_pos])
        {
           // printf(".");
            train_pos++;
            //	finaltestdatanum++;
            continue;
        }

				for(y=0;y<weight_data->num_output;y++)
				{
					if (weight_data->output[train_pos][y]==1 && train_classes_added[y]++>=maxtest)
					{

							
							addthis=0;
							continue;
					}
				}
				
				if(!addthis)
				{
					train_pos++;
					continue;
				}

      //  printf("x");

        
        for (y=0;y<weight_data->num_input;y++)
        {
            input[y]=weight_data->input[train_pos][y];
        }
        for (y=0;y<weight_data->num_output;y++)
        {
            if (weight_data->output[train_pos][y])
                output[y]=((weight_data->output[train_pos][y]));
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
        printf ( "neuro num\r\n" );
        exit ( 0 );
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
		
		unlink("active.net");

    printf("loading training data...");

    train_data = fann_read_train_from_file ( "bb-train-unscaled.dat" );
	printf(" train: %u",fann_length_train_data(train_data));
	
    test_data = fann_read_train_from_file ( "bb-test-unscaled.dat" );
		printf(" test: %u\r\n",fann_length_train_data(test_data));
    if (!train_data)
        exit(0);
		

    // fann_scale_train_data ( train_data, 0, 1.54 );
    //  fann_scale_train_data ( test_data, 0, 1.54 );
    weight_data=fann_merge_train_data(train_data,test_data);

    cln_weight_data=fann_duplicate_train_data(weight_data);
    cln_test_data=fann_duplicate_train_data(test_data);
    cln_train_data=fann_duplicate_train_data(train_data);

    //num_neurons_hidden = atoi ( argv[1] );



    unsigned num=0;
    unsigned y;
    unsigned u;
    unsigned x;
    unsigned reject;


    unsigned best_neur1,best_neur;
    best_neur1=((unsigned)fann_length_train_data(train_data)/
			(unsigned)train_data->num_input)/(unsigned)train_data->num_output-numn;
    best_neur=(cln_train_data->num_input+cln_train_data->num_output)/2;

    if (!ann)
    {
        if (!numn)
        {
            numn=3;
            l1n=best_neur;
            l2n=weight_data->num_output;
        }
        if (argc<6)
            l4n=weight_data->num_output;

        conn_rate=0.1f+((rand()%100)*0.01f);
        printf("create network: layers=%d l1n=%d l2n=%d l3n=%d l4n=%d\ l5n=%d l6n=%dr\n",numn,l1n,l2n,l3n,l4n,l5n,l6n);
        ann = fann_create_standard(//conn_rate,
                  numn,
                  weight_data->num_input,
                  l1n,
                  l2n,
                  l3n,
                  l4n,
                  l5n,
                  l6n,
                  train_data->num_output );
        if ( ( unsigned ) ann==NULL )
        {
            printf ( "error" );
            exit ( 0 );
        }

        unsigned mintraining=0;
//			for(int i=0;i<numn;i++)
        mintraining=(fann_get_total_neurons(ann)*(numn*4*2));

        printf ( "Creating normal network %p conn_rate: %.2f. minimum cases: %u, %.2f per class",ann ,conn_rate,mintraining,
                 ((double)mintraining*0.9/(double)weight_data->num_output));

        // fann_init_weights ( ann, train_data );

       // fann_set_activation_function_hidden ( ann,FANN_SIGMOID_SYMMETRIC);
   //     fann_set_activation_function_output ( ann,FANN_SIGMOID_SYMMETRIC );//FANN_SIGMOID_SYMMETRIC
			fann_set_activation_steepness_hidden(ann, 1);
			fann_set_activation_steepness_output(ann, 1);

        //	 fann_set_activation_function_layer(ann, FANN_SIGMOID_STEPWISE ,1);
        // fann_set_activation_function_layer(ann,FANN_ELLIOT ,2);
        // fann_set_activation_function_layer(ann,  FANN_GAUSSIAN_SYMMETRIC,3);


        //fann_set_activation_function_layer(ann,FANN_LINEAR_PIECE_SYMMETRIC,5);

        //	fann_set_activation_steepness_layer(ann, 0.65f, 1);
        //	fann_set_activation_steepness_layer(ann, 1.0f, 2);
        //fann_set_activation_steepness_layer(ann, 1.0f, 3);
        //fann_set_activation_steepness_layer(ann, 1.0f, 4);
        //fann_set_activation_steepness_layer(ann, 0.25f, 5);

        //	fann_set_activation_steepness_layer(ann, 1.0f, 2);

        //fann_set_activation_steepness_layer(ann, 1.0f, 1);

        // 	fann_set_bit_fail_limit(ann, 0.08f);



				fann_init_weights(ann, train_data);



        fann_set_training_algorithm ( ann, FANN_TRAIN_RPROP );


        //  fann_set_activation_steepness_layer(ann, 1.0f, 1);


       //  fann_randomize_weights ( ann, -1.0f, 1.0f );

        /*    if (fann_set_scaling_params(ann, train_data,-1.0f,1.0f,0.0f, 1.0f)==-1)
           printf("set scaling error\n");
        fann_scale_train(ann,train_data);
        //fann_scale_train(ann,weight_data);
        //    if (fann_set_scaling_params(ann, test_data,-1.0f,1.0f,-1.0f, 1.0f)==-1)
        //	printf("set scaling error\n");
        fann_scale_train(ann,test_data); */

        //	fann_set_rprop_increase_factor(ann, 1.3f);
    }
    else
    {
        //    fann_scale_train(ann,train_data);
        //   fann_scale_train(ann,weight_data);
//		fann_scale_train(ann,test_data);
        fann_set_training_algorithm ( ann, FANN_TRAIN_RPROP);


    }

    rebuild_functions();


    printf ( "input: %u, output: %u, neurons: %u best_neur1: %u best_neur: %u" ,
             weight_data->num_input, weight_data->num_output, num_neurons_hidden,
             best_neur1,best_neur);

//   fann_set_activation_steepness_layer(ann, 0.75f, 1);
    //fann_set_activation_steepness_layer(ann, 0.25f, 2);

    //  fann_set_train_error_function ( ann, FANN_ERRORFUNC_LINEAR );
    // fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
    nextalgo=fann_get_training_algorithm ( ann ) ;

    // fann_set_bit_fail_limit ( ann, ( fann_type ) 0.035f );

//   fann_set_callback ( ann, train_callback );

    //fann_scale_output_train_data(cln_train_data);
    //fann_scale_output_train_data(cln_test_data);

    //  fann_scale_train_data(test_data,-0.00843000, 0.01202000);
//   fann_reset_MSE ( ann );
    // fann_train_on_data ( ann, train_data, max_epochs, epochs_between_reports, desired_error );

    /*  printf ( "Testing network.\n" );

      test_data = fann_read_train_from_file ( "bb-train.test" );

      fann_reset_MSE ( ann );
      for ( i = 0; i < fann_length_train_data ( test_data ); i++ )
        {
          fann_test ( ann, test_data->input[i], test_data->output[i] );
        }

      printf ( "MSE error on test data: %f\n", fann_get_MSE ( ann ) );
      */
    // printf ( "\r\n\ttarget reached. saving network.\n" );


    classmin=fann_length_train_data(weight_data);
    printf("\r\ntrain classes: [");
    for ( y=0;y<weight_data->num_output;y++)
    {


        num=0;
        for (u=0;u<fann_length_train_data(weight_data);u++)
            if (weight_data->output[u][y]==1.0f)
                num++;
					//	else 
						//	printf("%.2f ",weight_data->output[u][y]);
        if (num<classmin)
            classmin=num;

        printf(" %u=%d ", y, num);
    }
		

    unsigned j,l;
    for ( j=0;j<fann_length_train_data(weight_data);j++)
    {
        reject=0;
        for ( x=0;x<weight_data->num_output;x++)
            if (weight_data->output[j][x]<=0.0f)
                reject++;
        //	else if(reject)
        //	printf("no %u ok - %f\n",reject,train_data->output[j][x]);
        if (reject>=weight_data->num_output)
        {
            //	printf(" rule %u: %.4f %.4f %.4f\n", j, train_data->output[j][0],
            //	train_data->output[j][1],
            //	train_data->output[j][2]);
            reject_total++;
        }
    }
    printf(" reject=%d", reject_total);
    printf(" ]");
	
		if(!num)
		{
			printf("zero data classes, classmin: %u\r\n",classmin);
			exit(0);
		}	

    classmin-=30;
    //classmin=100;
    finaldatanum=((unsigned)classmin*((unsigned)weight_data->num_output));


    printf ( "\r\ntrain_data=%p, count: %u input: %u, output: %u, neurons: %u bestneur: %u finaldata: %u classmin: %u left: %u",
             weight_data, fann_length_train_data(weight_data), weight_data->num_input, weight_data->num_output, num_neurons_hidden,
             best_neur,finaldatanum,classmin,fann_length_train_data(weight_data)-finaldatanum);




    for ( l=0;l<weight_data->num_output;l++)
        train_classes_added[l]=0;



    train_matrix = (unsigned*) malloc(sizeof(unsigned)*fann_length_train_data(weight_data)*2);

    for (l=0;l<fann_length_train_data(weight_data);l++)
        train_matrix[l]=1;

    printf("\r\ncreating %u train data %p ...", finaldatanum, train_matrix);

    final_data = fann_create_train_from_callback(finaldatanum, weight_data->num_input,
                 weight_data->num_output,
                 train_func);
unsigned classes[10];
memset(classes,sizeof(classes),0);
	for ( y=0;y<fann_length_train_data(final_data);y++)
	{
		
		char chars[]={'B','s','.'};
		num=0;
		for (u=0;u<final_data->num_output;u++)
		if (final_data->output[y][u]==1.0f)
		{
			classes[u]++;
			//printf("%c",chars[u]);
			num++;
		}
		// if (num<classmin)
		//   classmin=num;
		
		//printf(" %u=%d ", y, num);
	}
	printf("\r\nclasses [ ");
	for (i=0;i<final_data->num_output;i++)
	printf("%u=%u ",i,classes[i]);
	printf("] ");
	
    train_pos=0;

    finaltestdatanum=fann_length_train_data(weight_data)-finaldatanum;


		
    //for(l=0;l<weight_data->num_output;l++)
			//if(left_classes[l]<mintest)
				//mintest=left_classes[l];
				
    maxtest=mintest;
   // finaltestdatanum=maxtest*weight_data->num_output;
    printf("\r\ncreating %u test data [maxtest %u] ... ", finaltestdatanum,maxtest);

    for ( l=0;l<weight_data->num_output;l++)
        train_classes_added[l]=0;

    final_test_data = fann_create_train_from_callback(finaltestdatanum, weight_data->num_input,
                      weight_data->num_output,
                      test_train_func);

    
    for (i=0;i<10;i++)
        classes[i]=0;

    for ( y=0;y<fann_length_train_data(final_test_data);y++)
    {

        char chars[]={'B','s','.'};
        num=0;
        for (u=0;u<final_test_data->num_output;u++)
            if (final_test_data->output[y][u]==1.0f)
            {
                classes[u]++;
                //printf("%c",chars[u]);
                num++;
            }
        // if (num<classmin)
        //   classmin=num;

        //printf(" %u=%d ", y, num);
    }
    printf("\r\nclasses [ ");
    for (i=0;i<final_test_data->num_output;i++)
        printf("%u=%u ",i,classes[i]);
    printf("] ");

//	if (fann_set_scaling_params(ann, final_data,-1.0f,1.0f,0.0f, 1.0f)==-1)
    //printf("set scaling error: %s\n",fann_get_errno(ann));

    //   fann_scale_train_input(ann,train_data);
    // fann_scale_output_train_data(train_data,0.0f,1.0f);
    //   fann_scale_input_train_data(train_data, -1.0,0.0f);
    // fann_scale_output_train_data(test_data,-1.0f,1.0f);
    // fann_scale_input_train_data(test_data, -1.0,1.0f);
//	fann_scale_train(ann,final_data);
    //  fann_scale_train(ann,weight_data);
    //    fann_scale_train(ann,test_data);
    printf ( "ok\r\nsaving ...\n\r" );
    fann_save_train(final_data, "train.dat");
    fann_save_train(final_test_data, "test.dat");

    fann_save ( ann, "train.net" );

    printf ( "Cleaning up.\n" );

    

   // fann_destroy_train ( train_data );
   // fann_destroy_train ( test_data );
   // fann_destroy ( ann );

    return 0;
}

void rebuild_functions(void)
{
    int sygm_functions[]={FANN_SIGMOID_SYMMETRIC};
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

    int in_functions[]={FANN_SIGMOID_SYMMETRIC,FANN_SIGMOID_SYMMETRIC_STEPWISE};
    int out_functions[]={FANN_SIN_SYMMETRIC};//FANN_GAUSSIAN_SYMMETRIC,FANN_SIGMOID_SYMMETRIC,FANN_SIGMOID_SYMMETRIC_STEPWISE};

    unsigned l=1;
    numn=fann_get_num_layers(ann);
    printf("\r\n[ act funcs: ");
    for (;l<numn;l++)
    {
        int nfunc;
        if (l==1)
            nfunc=FANN_SIGMOID_SYMMETRIC_STEPWISE;//sygm_functions[rand()%((sizeof(sygm_functions)/sizeof(int)))];
        else if (l==numn-1)
            nfunc=FANN_SIGMOID_SYMMETRIC_STEPWISE;//out_functions[rand()%((sizeof(out_functions)/sizeof(int)))];
        else
            nfunc=sygm_functions[rand()%((sizeof(sygm_functions)/sizeof(int)))];
        //	printf("nfunc %u",nfunc);
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
            stp=1.0f;
        fann_set_activation_steepness_layer(ann, 	stp, l);
        fann_set_activation_function_layer(ann,nfunc,l);
        printf("\r\n #%-02d %s <%-4.02f l", FANN_ACTIVATIONFUNC_NAMES[	fann_get_activation_function(ann,l,0)],
               fann_get_activation_steepness(ann,l,0));


    }
    printf("]\r\n");
}