#include <stdio.h>
// #include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "matrixMultiplication.h"
#include "hiddenGateBias.h"
#include "hiddenGateWeights.h"
#include "inputGateBias.h"
#include "inputGateWeights.h"
#include "linearLayerbias.h"
#include "linearLayerWeight.h"
#include "input_0.h"
#include "input_5.h"
#include "input_10.h"
#include "input_15.h"
#include "input_20.h"
#include "input_25.h"
#include "input_30.h"
#include "input_35.h"
#include "input_40.h"
#include "y_df_0.h"
#include "y_df_5.h"
#include "y_df_10.h"
#include "y_df_15.h"
#include "y_df_20.h"
#include "y_df_25.h"
#include "y_df_30.h"
#include "y_df_35.h"
#include "y_df_40.h"
#include "golden.h"

#define VAR_104 104
#define VAR_48 48
#define VAR_52 52
#define VAR_96 96
#define FRAME_SIZE 5
#define SNR_SIZE 8




float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}


// Function to perform tanh activation on a vector
void tanhActivation(float vector[VAR_48], float result[VAR_48]) {
    for (int i = 0; i < VAR_48; i++) {
        result[i] = tanh(vector[i]);
    }
}

void resetGate(float Wir[VAR_48][VAR_104], float X[VAR_104], float bir[VAR_48], float Wt[VAR_48][VAR_48], float ht[VAR_48], float bt[VAR_48], float r[VAR_48]) {
    float r1[VAR_48], r2[VAR_48];
    for(int i=0;i<VAR_48;i++){
    	r1[i] = 0;
    	r2[i] = 0;
    }
    multiplyMatrixVector(Wir, X, r1);
    multiplyMatrixVector2(Wt, ht, r2);
    for(int i = 0; i < VAR_48; i++){
        r[i] = sigmoid(r1[i] + r2[i] + bir[i] + bt[i]);
    }
}

void updateGate(float Wiz[VAR_48][VAR_104], float X[VAR_104], float biz[VAR_48], float Wt[VAR_48][VAR_48], float ht[VAR_48], float bt[VAR_48], float z[VAR_48]) {
    float z1[VAR_48], z2[VAR_48];
    for(int i=0; i< VAR_48;i++){
    	z1[i]=0;
    	z2[i]=0;
    }
    multiplyMatrixVector(Wiz, X, z1);
    multiplyMatrixVector2(Wt, ht, z2);
    for(int i = 0; i < VAR_48; i++){
        z[i] = sigmoid(z1[i] + z2[i] + biz[i] + bt[i]);
    }
}

void tanh_layer(float Win[VAR_48][VAR_104], float X[VAR_104], float bin[VAR_48], float r[VAR_48], float Whn[VAR_48][VAR_48], float ht_1[VAR_48], float bh[VAR_48], float n[VAR_48]) {
    float n1[VAR_48], n2[VAR_48];
    for(int i=0; i< VAR_48;i++){
        	n1[i]=0;
        	n2[i]=0;
    }
    multiplyMatrixVector(Win, X, n1);
    multiplyMatrixVector2(Whn, ht_1, n2);
    for(int i = 0; i < VAR_48; i++){
        n[i] = (n2[i] + bh[i]) * r[i];
        n[i] += (n1[i] + bin[i]);
    }
    tanhActivation(n, n);
}

// DOUBT IN THIS FUNCTION
void ht_new(float z[VAR_48], float ht_1[VAR_48], float n[VAR_48], float ht[VAR_48]) {
    for(int i = 0; i < VAR_48; i++){
        ht[i] = (1 - z[i]) * n[i] + z[i] * ht_1[i];
    }
}

void linear_layer(float W[VAR_96][VAR_48], float n[VAR_48], float b[VAR_96], float y[VAR_96]) {
    multiplyMatrixVector3(W, n, y);
    for(int i = 0; i < VAR_96; i++){
        y[i] += b[i];
    }
}

float complex complexDiv(float complex a, float complex b){
//	printf("a=%f+I%f \n", crealf(a), cimagf(a));
//	printf("b=%f+I%f \n", crealf(b), cimagf(b));
	float mag = crealf(b)*crealf(b) + cimagf(b)*cimagf(b);
//	printf("mag=%f \n", mag);
	float complex out = (crealf(a)crealf(b) + cimagf(a)*cimagf(b) + I(-crealf(a)*cimagf(b)+crealf(b)*cimagf(a)))/mag;
//	printf("out=%f+%fI\n\n", crealf(out), cimagf(out));
	return out;

}
float complex complexMul(float complex a, float complex b){
	float complex out = (crealf(a)crealf(b) - cimagf(a)*cimagf(b) + I(crealf(a)*cimagf(b)+crealf(b)*cimagf(a)));
	return out;
}
void equalizer(float complex yd_real[VAR_48], float hprev[VAR_96], int i, float complex yeq[VAR_48]) {

    for(int i=0;i<VAR_48;i++){
    	yeq[i]=yd_real[i]/(hprev[i]+I*hprev[i+48]);
    }
}

void demapping(float complex yeq[VAR_48], float complex d[VAR_52], float complex yd[VAR_52], float complex hdpa[VAR_52]){
    int k=0;
    for(int i=0;i<VAR_52;i++){
        if(i==6 || i==20 || i==31){
            d[i] = 1;
        }
        else if(i==45){
            d[i] = -1;
        }
        else{
            if(crealf(yeq[k])>=0 && cimagf(yeq[k])>=0){
                d[i] = 1 + I;
            }
            else if(crealf(yeq[k])>=0 && cimagf(yeq[k])<=0){
                d[i] = 1 - I;
            }
            else if(crealf(yeq[k])<=0 && cimagf(yeq[k])>=0){
                d[i] = -1 + I;
            }
            else if(crealf(yeq[k])<=0 && cimagf(yeq[k])<=0){
                d[i] = -1 - I;
            }
            k++;
            d[i] = d[i] *0.707108;
        }

    }
    // ls estimation:
    for (int i=0;i<VAR_52;i++){
//    	hdpa[i]=complexDiv(yd[i],d[i]);
    	hdpa[i]=yd[i]/d[i];
    }

}

double GRU(float input[FRAME_SIZE][VAR_104],float complex y_df[FRAME_SIZE][100][VAR_52], float snr_val) {
    // Sample usage
    float ht[VAR_48], r[VAR_48];
    float z[VAR_48];
    float ht_1[VAR_48], n[VAR_48], output_gru[FRAME_SIZE][VAR_96];
    float complex yd_dsym[VAR_48]; // Data symbol in 96X1
    float complex yd[VAR_52]; // Data symbol in 52X1 complex values
    float complex d[VAR_52]; // Complex data nearest to the reference symbol (Demodulated Output)
    float complex hDPA_complex[VAR_52];
    float hLS[VAR_104], hLS_D[VAR_96], hDPA[VAR_104];
    float hprev[FRAME_SIZE][100][VAR_104]={0};
    float complex yeq[VAR_48];
    int iter;
    double complex temp[VAR_48];
    double temp4[48], temp5[48];

//    for(int i=0;i<VAR_48;i++){
//    	ht[i]=0;
//    	ht_1[i]=0;
//    }

//    for(int i = 0; i < VAR_104; i++){
//        hLS[i]=0;
//        hDPA[i] = 0;
//    }
//    for(int i=0;i<VAR_52;i++){
//    	d[i] = 1+I;
//    	yd[i] = 0+0*I;
//    }
//
//    for(int i=0;i<VAR_48;i++){
//    	yd_dsym[i] = 0;
//    }

    // Call the functions
    double nmse_sum_frame = 0;
    double mse_sum_iter = 0, nmse_sum=0, hpf=0;
    for(int i = 0; i < FRAME_SIZE; i++){
        for(int i=0;i<VAR_48;i++){
        	ht[i]=0;
        	ht_1[i]=0;
        }
        for(int i = 0; i < VAR_104; i++){
            hLS[i]=0;
            hDPA[i] = 0;
        }
        for(int i=0;i<VAR_52;i++){
        	d[i] = 1+I;
        	yd[i] = 0+0*I;
        }

        for(int i=0;i<VAR_48;i++){
        	yd_dsym[i] = 0;
        }
        int k = 0;
		for (int j = 0; j < VAR_104; j++) {
			if (j != 6 && j != 20 && j != 31 && j != 45 && j != 58 && j != 72 && j != 83 && j != 97) {
				hLS_D[k] = input[i][j];
				k++;
			}
			hLS[j] = input[i][j];
		}


        for(iter=0;iter<100;iter++){
    		k=0;
    		for(int j=0;j<VAR_52;j++){
    			yd[j] = y_df[i][iter][j];
    			hDPA_complex[j] = 0;
    		}

            for(int j=0;j<VAR_52;j++){
            	if(j==6 || j==20 || j==31 || j==45){
            		continue;
            	}
            	else{
            		yd_dsym[k] = crealf(y_df[i][iter][j]) + I*cimagf(y_df[i][iter][j]);
            		k++;
            	}
            }
            k=0;

            if(iter==0){
				resetGate(rgiw, hLS, rgib, rghw, ht_1, rghb, r);
				updateGate(ugiw, hLS, ugib, ughw, ht_1, ughb, z);
				tanh_layer(tgiw, hLS, tgib, r, tghw, ht_1, tghb, n);
				ht_new(z, ht_1, n, ht);
				linear_layer(outWeight, ht, outBias, output_gru[i]);

				for(k=0;k<VAR_48;k++){
					ht_1[k] = ht[k];
				}

				k=0;
	            equalizer(yd_dsym, output_gru[i], i, yeq); // Gives yEqualized output of size 48X1 Complex values

	            demapping(yeq, d, yd, hDPA_complex); // hDPA_complex is now updated


	            for(int j=0;j<VAR_52;j++){
	            	hDPA[j] = (float)crealf(hDPA_complex[j]);
	                hDPA[j+52] = (float)cimagf(hDPA_complex[j]);
	            }

	            // Time Averaging
	            // printf("Hprev for FRAME = %d\n",i);
	           	// printf("iter=%d\n", iter);
	            for(int j=0;j<VAR_104;j++){
	            	hprev[i][iter][j] = 0.5*hLS[j] + 0.5*hDPA[j];
		            // printf("%f, ", hprev[i][iter][j]);
	            }
            }
            else{
				resetGate(rgiw, hprev[i][iter-1], rgib, rghw, ht_1, rghb, r);
//				for(int q=0;q<VAR_48;q++){
//					printf("%f, ", r[q]);
//				}
				updateGate(ugiw, hprev[i][iter-1], ugib, ughw, ht_1, ughb, z);
				tanh_layer(tgiw, hprev[i][iter-1], tgib, r, tghw, ht_1, tghb, n);
				ht_new(z, ht_1, n, ht);
				linear_layer(outWeight, ht, outBias, output_gru[i]);
				for(k=0;k<VAR_48;k++){
					ht_1[k] = ht[k];
				}
	            equalizer(yd_dsym, output_gru[i], i, yeq); // Gives yEqualized output of size 48X1 Complex values - tested and works like a charm

	            demapping(yeq, d, yd, hDPA_complex); // hDPA_complex is now updated


	            for(int j=0;j<VAR_52;j++){
	            	hDPA[j] = (float)crealf(hDPA_complex[j]);
	                hDPA[j+52] = (float)cimagf(hDPA_complex[j]);
	             }
	            // Time Averaging
	            // printf("Hprev for FRAME = %d\n",i);
	            // printf("iter=%d\n", iter);
	            for(int j=0;j<VAR_104;j++){
	            	hprev[i][iter][j] = 0.5*hDPA[j] + 0.5*hprev[i][iter-1][j];
	            	// printf("%f, ",hprev[i][iter][j]);
	            }
	            // printf("\n\n");
            }

        }
        for(int mse_temp=0;mse_temp<100;mse_temp++){
        	double verif1[48], verif2[48];
        	double complex verify_comp[VAR_48];

        	for(int q=0;q<48;q++){
				temp4[q] = 0;
				temp5[q] = 0;
			}
        	for(int q=0;q<48;q++){
        		temp[q] = 0;
        	}
			int f = 0;
			for(int q=0;q<52;q++){
				if (q != 6 && q != 20 && q != 31 && q != 45 && q != 58 && q != 72 && q != 83 && q != 97) {
					temp4[f] = hprev[i][mse_temp][q];
					f++;
				}
			}
			f = 0;
			for(int q=52;q<104;q++){
				if (q != 6 && q != 20 && q != 31 && q != 45 && q != 58 && q != 72 && q != 83 && q != 97) {
					temp5[f] = hprev[i][mse_temp][q];
					f++;
				}
			}
			f = 0;
			for(int q=0;q<48;q++){
				temp[q] = temp4[q] + I*temp5[q];
			}
			f = 0;
			for(int q=0;q<48;q++){
				verif1[q] = verify[i][mse_temp][q];
			}
			f = 0;
			for(int q=0;q<48;q++){
				verif2[q] = verify[i][mse_temp][q+48];
			}
			for(int q=0;q<48;q++){
				verify_comp[q] = verif1[q] + I*verif2[q];
			}

			for(int q=0;q<48;q++){
				double complex temp2 = verify_comp[q];
				double complex temp3 = verify_comp[q] - temp[q];
				hpf += cabsf(temp2)*cabsf(temp2);
				mse_sum_iter += cabsf(temp3)*cabsf(temp3);
			}
        }
    }
    nmse_sum_frame = mse_sum_iter/hpf;
    return nmse_sum_frame;
}

int main()
{
   double mse_0 = GRU(input_0, y_df_0, 0);
       printf("mse_0 = %f\n", mse_0);

   double mse_5 = GRU(input_5, y_df_5, 5);
       printf("mse_5 = %f\n", mse_5);

   double mse_10 = GRU(input_10, y_df_10, 10);
       printf("mse_10 = %f\n", mse_10);

   double mse_15 = GRU(input_15, y_df_15, 15);
       printf("mse_15 = %f\n", mse_15);

   double mse_20 = GRU(input_20, y_df_20, 20);
       printf("mse_20 = %f\n", mse_20);

   double mse_25 = GRU(input_25, y_df_25, 25);
       printf("mse_25 = %f\n", mse_25);

   double mse_30 = GRU(input_30, y_df_30, 30);
       printf("mse_30 = %f\n", mse_30);

   double mse_35 = GRU(input_35, y_df_35, 35);
       printf("mse_35 = %f\n", mse_35);

   double mse_40 = GRU(input_40, y_df_40, 40);
//    printf("NMSE:  SNAR_0 = 0.29665, SNR_5 = 0.093882, SNR_10 = 0.02941,SNR_15 = 0.00966, SNR_20 = 0.003298, SNR_25 = 0.001310, SNR_30 = 0.000704, SNR_35 = 0.000496, SNR_40 = 0.000433\n");

    printf("mse_40 = %f", mse_40);
    return 0;
}
