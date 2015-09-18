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

void rebuild_functions(void);

int FANN_API train_callback
( struct fann *ann, struct fann_train_data *train,
  unsigned int max_epochs, unsigned int epochs_between_reports,
  double desired_error, unsigned int cur_epochs )
{
    if (stagn_epoch<0||stagn_epoch>1000)
        stagn_epoch=0;


    //  fann_reset_MSE(ann);

    mse_train = fann_test_data ( ann, train );
    bit_fail_train = fann_get_bit_fail ( ann );
    //  		fann_test ( ann, train_data->input[0], train_data->output[0] );
    //mse_train=fann_get_MSE(ann);
    //fann_reset_MSE(ann);
    mse_test = fann_test_data ( ann, test_data );
    bit_fail_test = fann_get_bit_fail ( ann );


    weight_mse=fann_test_data(ann,weight_data);


    if (prev_mse-mse_train>0)
    {
        prev_mse_chg[cur_mse_chg]=prev_mse-mse_train;
        if (cur_mse_chg++>=60)
            cur_mse_chg=0;
    }

    if (time(NULL)-lastmsechecktime>=1)
    {
        if (time(NULL)-last_min_timeleft_upd>=1*60)
        {
            last_min_timeleft_upd=time(NULL);
            last_min_timeleft=minutes_left;
        }

        int i;
        lastmsechecktime=time(NULL);

        double mse_min=0;

        for ( i=0;i<60;i++)
            mse_chg+=prev_mse_chg[i];

        //mse_chg=mse_chg;
        mse_min=mse_chg*60.0f;


        double mse_left=0;
        mse_left=(mse_train-desired_error);
        if (mse_min>0)
        {
            double prev_m;
            prev_m=mse_left/mse_min;

            minutes_left= (prev_m/60.0f);
        }
        mse_chg=0;
    }
    /*
        if (epochs-lowest_test_mse_epoch>155000)
        {
            fprintf(stderr,"\r\nreset - train failed\r\n");

            // return -1;
            fann_set_activation_function_hidden ( ann, FANN_SIGMOID_SYMMETRIC);
            fann_set_activation_function_output ( ann,  FANN_SIGMOID_SYMMETRIC);

            fann_set_training_algorithm ( ann, FANN_TRAIN_SARPROP );
            fann_set_train_error_function ( ann, FANN_ERRORFUNC_TANH );

            fann_set_learning_rate ( ann, 0.7f );
            fann_init_weights ( ann, train_data );

            stagn_epoch=0;

            lowest_test_mse_epoch=epochs;

            return;
        }
    */
    if (mse_test<min_mse_test)
    {
        stagn_epoch=0;
        fann_save ( ann, "bb-normal.net" );
        lowest_test_mse_epoch=cur_epochs;
        min_mse_test=mse_test;


    }

    if (mse_train<min_mse_train)
    {

        fann_save ( ann, "bb-normal-train.net" );
        min_mse_train=mse_train;
    }

    //if (mse_test>=prev_mse_test)
    //{
    //  stagn_epoch=(stagn_epoch?stagn_epoch:0.1)*1.35;
    //}

//   fann_set_learning_momentum ( ann, ( min ( mse_test, mse_train ) / max ( mse_test, mse_train ) ) *1.0f );

    /* if ( prev_mse - mse_test < 0.01 && last_bads++>=4 )
     {

         do
         {
             //func_num=func_num+rand()%4;
             //fann_set_training_algorithm(ann, func_num);
             if ( fann_get_learning_rate ( ann ) <=1.0 )
             {
                 fann_set_learning_rate ( ann, fann_get_learning_rate ( ann ) +0.01f );
                 printf ( "  *  inc learn ratio " );
             }
             else if ( fann_get_train_error_function ( ann ) !=FANN_ERRORFUNC_TANH )
             {
                 fann_set_train_error_function ( ann, FANN_ERRORFUNC_TANH );
                 printf ( "   * set train error func TANH" );
             }
             else
             {
                 fann_set_learning_momentum ( ann, ( min ( mse_test, mse_train ) / max ( mse_test, mse_train ) ) *1.0f );
                 printf ( "   * change learning_momentum %.4f",fann_get_learning_momentum ( ann ) );
             }

         }
         while ( fann_get_errno ( ( struct fann_error* ) ann ) == 12 );

         last_bads=0;
         func_num=0;
     }
     else if ( last_bads>=4 && prev_mse - mse_test>  0.01 )
     {
         last_bads--;
         fann_set_learning_rate ( ann, fann_get_learning_rate ( ann )-0.01f );
         fann_set_train_error_function ( ann, FANN_ERRORFUNC_LINEAR );
         printf ( "   * set train error func LINEAR, learn rate %.4f", fann_get_learning_rate ( ann ) );
     }*/
    /*                                               */
    double rinc;
    rinc=1.0f+((rand()%200)*0.01f);

    double rfactors[] = {1.2f,1.3f};

    rinc=(rand()%(sizeof(rfactors)/sizeof(double)-1));
    rinc=rfactors[(int)rinc];
//fann_set_rprop_increase_factor(ann, rinc);




    double l;
    l=(rand()) % 10;
    //fann_get_learning_rate ( ann )-
    //fann_set_learning_rate ( ann, 0.1f+(l*0.1) );


    /*                                               */
    prev_mse_test=mse_test;

    //	if(prevbitfail<=bit_fail_train)
    //		stagn_epoch=(stagn_epoch?stagn_epoch:0.1)*0.05;
    if (mse_train<prev_mse)
        stagn_epoch=(stagn_epoch>=0.01?stagn_epoch:0.1)*0.95;
    else  if (mse_train>=prev_mse)
        stagn_epoch=(stagn_epoch>=0.01?stagn_epoch:0.1)*1.15;

    if (stagn_epoch>=0.04f&&cur_epochs-stpns_epoch>5000)
    {


        double stepnesses[] = {0.15f,0.5f,0.01f,0.25f,0.50f,0.75f,1.0f};

        stpns=(rand()%(sizeof(stepnesses)/sizeof(double)-1));


        stpns=stepnesses[(int)stpns];
        int nl;
        nl=1+rand()%(fann_get_num_layers(ann));

        printf("* stepness %0.2f l %d",stpns,nl);

        if (nl!=0)
            fann_set_activation_steepness_layer(ann, 	stpns, nl);
        // fann_set_activation_steepness_layer(ann, stpns, 1);
        stpns_epoch=cur_epochs;
        stagn_epoch=stagn_epoch*0.2;
    }
    else
    {
        //	printf(" %d",epochs-stpns_epoch);
    }
    double rdec;
    if ((mse_train>=prev_mse||mse_train>0.01f||mse_test>0.01f) &&
            ((cur_epochs>1055) &&(stagn_epoch>=0.1f)))
    {
        rdec=(rand()%100)*0.001f;
        if (!rdec)
            rdec=0.01f;
        // fann_set_activation_function_hidden ( ann,  rand()*0.81);
        printf(" * rprop_increase_factor %.2f",rdec);
        //	rebuild_functions();

        // fann_set_rprop_increase_factor(ann, rinc);



        //fann_set_training_algorithm ( ann, nextalgo);//);

      //  fann_set_learning_rate ( ann, (rand()%10)*0.1f);//0.9f );

        //		fann_set_rprop_decrease_factor(ann, rdec);

        stagn_epoch=stagn_epoch*0.2;

        prevsarep=cur_epochs;
    }
    if (nextalgo++>=4)
        nextalgo=0;
    if (cur_epochs>555&&stagn_epoch>=1.0f)
    {
        //	fann_set_activation_function_hidden ( ann,  FANN_SIGMOID_SYMMETRIC_STEPWISE);
        nextalgo=(rand()%1) ? FANN_TRAIN_QUICKPROP : FANN_TRAIN_SARPROP;
        // fann_set_training_algorithm ( ann, nextalgo );
        stagn_epoch=0;
    }

    prevbitfail=bit_fail_train;
    prev_mse = mse_train;

    fann_type *calc_out,*calc_out2;
    unsigned calc1,calc2;

    calc1=fann_length_train_data(test_data)-100;
    calc2=rand()%fann_length_train_data(train_data);
    printf(" cal1 %u cal2 %u",calc1,calc2);
    //fann_scale_input(ann, test_data->input[calc1]);
    //fann_scale_input(ann, train_data->input[calc2]);
    //	fann_scale_output(ann, test_data->input[calc1]);
    calc_out = fann_run(ann, test_data->input[calc1]);
    fann_descale_output(ann, calc_out);
    double val_1[10];
    memcpy(&val_1,calc_out, sizeof(double)*3);
    //fann_scale_input(ann, train_data->input[calc2]);
    calc_out2 = fann_run(ann, train_data->input[calc2]);
    fann_descale_output(ann,calc_out2);
    double val_2[10];
    memcpy(&val_2,  calc_out2, sizeof(double)*3);

    //fann_descale_input(ann, train_data->input[calc2]);
    //fann_descale_input(ann, test_data->input[calc1]);
    //calc_out = fann_test(ann, test_data->input[calc1], test_data->output[calc1]);
//	calc_out2 = fann_test(ann, train_data->input[calc2], train_data->output[calc2]) ;





//%s lr %.02f stpns %.2f stg %.2f rinc %.2f rdec %.2f hleft %.2f
    /* FANN_TRAIN_NAMES[ fann_get_training_algorithm ( ann ) ],
    fann_get_learning_rate ( ann ),
    fann_get_activation_steepness(ann, 1, 1) ,
    stagn_epoch,
    fann_get_rprop_increase_factor(ann),
    fann_get_rprop_decrease_factor(ann),
    minutes_left,*/
    printf
    ( "\r\n>epoch %-5u neur %-3d trmse %.06f (%.08f) tstmse %.06f (%.08f e=%-3d) %.04f trnbf %-5d  tstbf %-5d\r\n[%7.4f:%7.4f %7.4f:%7.4f %7.4f:%7.4f | %7.4f:%7.4f %7.4f:%7.4f %7.4f:%7.4f]",
      cur_epochs, ann->total_neurons, mse_train, min_mse_train, mse_test, min_mse_test, lowest_test_mse_epoch,
      weight_mse,			bit_fail_train,
      bit_fail_test,

      val_2[0],
      cln_train_data->output[calc2][0],
      val_2[1],
      cln_train_data->output[calc2][1],
      val_2[2],
      cln_train_data->output[calc2][2],
      val_1[0],
      cln_test_data->output[calc1][0],
      val_1[1],
      cln_test_data->output[calc1][1],
      val_1[2],
      cln_test_data->output[calc1][2]
    );

//fann_reset_MSE(ann);

    return 1;

}
void sig_term ( int p )
{
    printf ( "\r\nsaving net...\r\n" );
    fann_save ( ann, "bb-normal.net" );
    exit ( 0 );
}

int main ( int argc, char **argv )
{

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

    train_data = fann_read_train_from_file ( "bb-train-unscaled.dat" );
    test_data = fann_read_train_from_file ( "bb-test-unscaled.dat" );
    // fann_scale_train_data ( train_data, 0, 1.54 );
    //  fann_scale_train_data ( test_data, 0, 1.54 );
    weight_data=fann_merge_train_data(train_data,test_data);

    cln_weight_data=fann_duplicate_train_data(weight_data);
    cln_test_data=fann_duplicate_train_data(test_data);
    cln_train_data=fann_duplicate_train_data(train_data);

    //num_neurons_hidden = atoi ( argv[1] );

    if ( ( ann = fann_create_from_file ( "bb-normal.net" ) ) !=NULL )
    {
        printf ( "Loaded normal network %p.\n",ann );
        //fann_print_connections(ann);
        if (num_neurons_hidden == -1)
        {
            printf("reset, numo %d\r\n",ann->num_output);
            unlink("bb-normal.net");
            fann_destroy(ann);
            ann=NULL;
            exit(0);
        }
    }

    int num=0;
    int y;
    int u;
    int x;
    unsigned reject;


    printf("classes [ ");
    for (y=0;y<train_data->num_output;y++)
    {


        num=0;
        for (u=0;u<fann_length_train_data(train_data);u++)
            if (train_data->output[u][y]==1.0f)
                num++;

        printf(" %d=%d ", y, num);
    }

    int j;
    int reject_total=0;

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



    int best_neur;
    best_neur=fann_length_train_data(train_data)/train_data->num_input/train_data->num_output-numn;

    if (!ann)
    {
        if (!numn)
        {
            numn=3;
            l1n=best_neur;
            l2n=train_data->num_output;
        }
        if (argc<6)
            l4n=train_data->num_output;

        printf("create network: layers=%d l1n=%d l2n=%d l3n=%d l4n=%d\ l5n=%d l6n=%dr\n",numn,l1n,l2n,l3n,l4n,l5n,l6n);
        ann = fann_create_standard ( numn,
                                     train_data->num_input,
                                     l1n,
                                     l2n,
                                     l3n,
                                     l4n,
                                     l5n,
                                     l6n,
                                     train_data->num_output );
        if ( ( int ) ann==NULL )
        {
            printf ( "error" );
            exit ( 0 );
        }

        int mintraining=0;
//			for(int i=0;i<numn;i++)
        mintraining=(fann_get_total_neurons(ann)*(numn*4*2));

        printf ( "Creating normal network %p. minimum cases: %u, %.2f per class",ann ,mintraining,
                 ((double)mintraining*0.9/(double)train_data->num_output));

        //	fann_init_weights ( ann, train_data );




        fann_set_activation_function_hidden ( ann,FANN_SIGMOID_SYMMETRIC);
        fann_set_activation_function_output ( ann,FANN_LINEAR_PIECE_SYMMETRIC );


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






        if (fann_set_scaling_params(ann, train_data,-1.0f,1.8f,0.0f, 1.0f)==-1)
            printf("set scaling error: %s\n",fann_get_errno(ann));

        //   fann_scale_train_input(ann,train_data);
        // fann_scale_output_train_data(train_data,0.0f,1.0f);
        //   fann_scale_input_train_data(train_data, -1.0,0.0f);
        // fann_scale_output_train_data(test_data,-1.0f,1.0f);
        // fann_scale_input_train_data(test_data, -1.0,1.0f);
        fann_scale_train(ann,train_data);
        fann_scale_train(ann,weight_data);
        //     fann_scale_train(ann,test_data);
        fann_set_training_algorithm ( ann, FANN_TRAIN_RPROP );


        //  fann_set_activation_steepness_layer(ann, 1.0f, 1);


        //	fann_randomize_weights ( ann, -0.4f, 0.4f );

        fann_init_weights ( ann, train_data );
		rebuild_functions();
        //	fann_set_rprop_increase_factor(ann, 1.3f);
    }
    else
    {
        fann_scale_train(ann,train_data);
        fann_scale_train(ann,weight_data);
//		fann_scale_train(ann,test_data);
        fann_set_training_algorithm ( ann, FANN_TRAIN_RPROP);


    }

    


    printf ( "train_data=%p, count: %d input: %d, output: %d, neurons: %d bestneur: %d\n",
             train_data, fann_length_train_data(train_data), train_data->num_input, train_data->num_output, num_neurons_hidden,
             best_neur);

//   fann_set_activation_steepness_layer(ann, 0.75f, 1);
    //fann_set_activation_steepness_layer(ann, 0.25f, 2);

    fann_set_train_error_function ( ann, FANN_ERRORFUNC_LINEAR );
    fann_set_train_stop_function(ann, FANN_STOPFUNC_MSE);
    nextalgo=fann_get_training_algorithm ( ann ) ;

    fann_set_bit_fail_limit ( ann, ( fann_type ) 0.035f );

    fann_set_callback ( ann, train_callback );

    //fann_scale_output_train_data(cln_train_data);
    //fann_scale_output_train_data(cln_test_data);

    //  fann_scale_train_data(test_data,-0.00843000, 0.01202000);
    fann_reset_MSE ( ann );
    fann_train_on_data ( ann, train_data, max_epochs, epochs_between_reports, desired_error );

    /*  printf ( "Testing network.\n" );

      test_data = fann_read_train_from_file ( "bb-train.test" );

      fann_reset_MSE ( ann );
      for ( i = 0; i < fann_length_train_data ( test_data ); i++ )
        {
          fann_test ( ann, test_data->input[i], test_data->output[i] );
        }

      printf ( "MSE error on test data: %f\n", fann_get_MSE ( ann ) );
      */
    printf ( "\r\n\ttarget reached. saving network.\n" );

    fann_save ( ann, "fann_normal.net" );

    printf ( "Cleaning up.\n" );
    fann_destroy_train ( train_data );
    fann_destroy_train ( test_data );
    fann_destroy ( ann );

    return 0;
}

void rebuild_functions(void)
{
    int l=1;
    printf("\r\n[ act funcs: ");
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

        fann_set_activation_steepness_layer(ann, 	0.1+(stp*0.01), l);
        fann_set_activation_function_layer(ann,nfunc,l);
        printf("<lay#%-02d %s:%-4.02f> ", l, FANN_ACTIVATIONFUNC_NAMES[	fann_get_activation_function(ann,l,0)],
               fann_get_activation_steepness(ann,l,0));


    }
    printf("]\r\n");
}
