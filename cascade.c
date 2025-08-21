#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <memory.h>
#include <unistd.h>
#include "fann_common.h"

struct fann_train_data *train_data, *test_data,*cln_train_data;
struct fann *ann;
const float desired_error = ( const float ) 0.0001f;
unsigned int max_neurons = 6000;
unsigned int neurons_between_reports = 1;
unsigned int bit_fail_train, bit_fail_test;
float mse_train=0, mse_test=0, prev_mse=0, min_mse_train=1, min_mse_test=1;
unsigned int i = 0;
fann_type *output;
fann_type steepness[5];
int multi = 0;
int last_bads = 0;
enum fann_activationfunc_enum activation[7];
enum fann_train_enum training_algorithm = FANN_TRAIN_RPROP;
int func_num=0;
int lowest_test_mse_epoch=0;
double jitter_factor=0.001f;
double test_perc,train_perc;
double jitt_value;

char histfile[]="cascade_hist.dat";
void plot(double p1, double p2,double p3,double p4,double p5)
{
    FILE *f;
    f=fopen(histfile, "a");
    char str[128];
    sprintf(str,"%f %.8f %.8f %.8f %.8f\n",p1,p2,p3,p4,p5);
    fwrite(str, strlen(str),1,f);
    fclose(f);
}

void jitter_train(struct fann_train_data *data, struct fann_train_data *clean_data)
{
    int i;
    int inc;
    inc=rand()%2;
    for (i=0;i<fann_length_train_data(clean_data);i++)
    {
        int x;

        for (x=0;x<data->num_input;x++)
        {

            jitt_value=((rand()%1000)*jitter_factor);
            if (inc)
                data->input[i][x]=clean_data->input[i][x]-jitt_value;
            else
                data->input[i][x]=clean_data->input[i][x]+jitt_value;
        }
    }
}
int ftest_data(void)
{

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
        calc_out2 = fann_run(ann, train_data->input[calc2]);
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
    train_perc=(success/fann_length_train_data(train_data))*100.0f;

    //fails,success,perc
    //);

    fails=0;
    success=0;

    for (curi=0;curi<fann_length_train_data(test_data);curi++)
    {

        calc2=curi;//rand()%(fann_length_train_data(train_data)-1);
        calc_out2 = fann_run(ann, test_data->input[calc2]);
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
    test_perc=(success/fann_length_train_data(test_data))*100.0f;

    //fails,success,perc
    //);
}
int FANN_API cascade_callback
( struct fann *ann, struct fann_train_data *train,
  unsigned int max_epochs, unsigned int epochs_between_reports,
  float desired_error, unsigned int epochs )
{

    mse_train = fann_test_data ( ann, train_data );
    bit_fail_train = fann_get_bit_fail ( ann );
    mse_test = fann_test_data ( ann, test_data );
    bit_fail_test = fann_get_bit_fail ( ann );
    if (mse_test<min_mse_test)
    {
        fann_save ( ann, "cascaded-test.net" );
        min_mse_test=mse_test;
        lowest_test_mse_epoch=epochs;
    }

    if (mse_train<min_mse_train)
    {
        fann_save ( ann, "cascaded.net" );
        min_mse_train=mse_train;
    }

    plot((double)epochs,mse_train,mse_test,train_perc/100,test_perc/100);
    // {

    // do
    // {

    // activation[0] = ( enum fann_activationfunc_enum ) func_num;
    // }
    // }
    // else if ( last_bads>=1 && prev_mse > mse_test )
    // last_bads--;
    ftest_data();
    printf
    ( "\n %5d %4d %.08f %5.2f%% (%.08f) | %.08f %5.2f%% (%.08f e=%d) | %-4d  %-4d %.2lf %s",
      epochs, ann->total_neurons, mse_train,train_perc, min_mse_train, mse_test, test_perc,min_mse_test, lowest_test_mse_epoch, bit_fail_train,
      bit_fail_test,
      ( ann->last_layer - 2 )->first_neuron->activation_steepness,
      FANN_ACTIVATIONFUNC_NAMES[ ( ann->last_layer -
                                   2 )->first_neuron->activation_function] );
    jitter_train(train, cln_train_data);
    return 0;

}
void sig_term ( int p )
{
    printf ( "\r\nsaving net...\r\n" );

    exit ( 0 );
};
int main(int argc,char **argv)
{
    unlink(histfile);
    srand ( time ( NULL ) );

    train_data = fann_read_train_from_file ( "train.dat" );
    test_data = fann_read_train_from_file ( "test.dat" );

    cln_train_data=fann_duplicate_train_data(train_data);
    printf ( "Creating cascaded network.\n" );
    ann =
        fann_create_shortcut ( 2, fann_num_input_train_data ( train_data ),
                               fann_num_output_train_data ( train_data ) );
    fann_set_training_algorithm ( ann, FANN_TRAIN_RPROP );
    fann_set_activation_function_hidden ( ann, FANN_SIGMOID );
    fann_set_activation_function_output ( ann, FANN_SIGMOID);
    fann_set_train_error_function ( ann, FANN_ERRORFUNC_LINEAR );
    /*
     * fann_set_cascade_output_change_fraction(ann, 0.1f);
     *  ;
     * fann_set_cascade_candidate_change_fraction(ann, 0.1f);
     *
     */

    fann_set_callback ( ann, cascade_callback );
    if ( !multi )
    {

        /*  */
        //  steepness[0] = 0.22;
        steepness[0] = 0.9;
        steepness[1] = 1.0;

        /*
         * steepness[1] = 0.55;
         *  ;
         * steepness[1] = 0.33;
         *  ;
         * steepness[3] = 0.11;
         *  ;
         * steepness[1] = 0.01;
         *
         */

        /*
         *  steepness = 0.5;
         *
         */
        /*
         * activation = FANN_SIN_SYMMETRIC;
         */

        /*
         * activation[0] = FANN_SIGMOID;
         *
         */
        activation[0] = FANN_SIGMOID;

        /*
         * activation[2] = FANN_ELLIOT_SYMMETRIC;
         *
         */
        activation[1] = FANN_LINEAR_PIECE;

        /*
         * activation[4] = FANN_GAUSSIAN_SYMMETRIC;
         *  ;
         * activation[5] = FANN_SIGMOID;
         *
         */
        activation[2] = FANN_ELLIOT;
        activation[3] = FANN_COS;
        /*
         *
         *
         */
        activation[4] = FANN_SIN;
        fann_set_cascade_activation_functions ( ann, activation, 5);
        /*   fann_set_cascade_num_candidate_groups ( ann,
                                                  fann_num_input_train_data
                                                  ( train_data ) ); */

    }
    else
    {

        /*
         * fann_set_cascade_activation_steepnesses(ann, &steepness, 0.75);
         *
         */
    }

    /* TODO: weight mult > 0.01 */
    /*  if ( training_algorithm == FANN_TRAIN_QUICKPROP )
      {
          fann_set_learning_rate ( ann, 0.35f );
      }
      else
      {
          fann_set_learning_rate ( ann, 0.7f );

      }
      fann_set_bit_fail_limit ( ann, ( fann_type ) 0.9f );*/

    /*
     * fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
     *
     */

    fann_init_weights ( ann, train_data );

    printf ( "Training network.\n" );
    fann_cascadetrain_on_data ( ann, train_data, max_neurons,
                                1, desired_error );
    fann_print_connections ( ann );
    mse_train = fann_test_data ( ann, train_data );
    bit_fail_train = fann_get_bit_fail ( ann );
    mse_test = fann_test_data ( ann, test_data );
    bit_fail_test = fann_get_bit_fail ( ann );
    printf
    ( "\nTrain error: %.08f, Train bit-fail: %d, Test error: %.08f, Test bit-fail: %d\n\n",
      mse_train, bit_fail_train, mse_test, bit_fail_test );

    printf ( "Saving cascaded network.\n" );
    fann_save ( ann, "cascaded.net" );

    fann_destroy_train ( train_data );
    fann_destroy_train ( test_data );
    fann_destroy ( ann );
    return 0;

}
