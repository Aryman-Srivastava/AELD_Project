/******************************************************************************
*
* Copyright (C) 2009 - 2014 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

/*
 * helloworld.c: simple test application
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include <xtime_l.h>
#include "xaxidma.h"
#include <stdio.h>
 #include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "xparameters.h"
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
#define VAR_152 152
#define VAR_144 144
#define FRAME_SIZE 5
#define SNR_SIZE 8


// Define the piecewise linear function
double sigmoid(float x) {
    float parameters[][3] = {
    {0.0078125, 0.05, -3.4},
    {0.0625, 0.24, -1.3},
    {0.25, 0.5, 1.3},
    {0.0625, 0.76, 3.4},
    {0.0078125, 0.95, INFINITY}};
    int i;
    for (i = 0; i < sizeof(parameters) / sizeof(parameters[0]); i++) {
        double m = parameters[i][0];
        double k = parameters[i][1];
        double n = parameters[i][2];
        if (x <= n) {
            return m * x + k;
        }
    }
    // Return 0 for values of x greater than the last interval
    return 0.0;
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
	float complex out = (crealf(a)*crealf(b) + cimagf(a)*cimagf(b) + I*(-crealf(a)*cimagf(b)+crealf(b)*cimagf(a)))/mag;
//	printf("out=%f+%fI\n\n", crealf(out), cimagf(out));
	return out;

}

float complex complexMul(float complex a, float complex b){
	float complex out = (crealf(a)*crealf(b) - cimagf(a)*cimagf(b) + I*(crealf(a)*cimagf(b)+crealf(b)*cimagf(a)));
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

int main_PS()
{

  XTime time_PS_start , time_PS_end;
  float curr_time;
  XTime_SetTime(0);
  XTime_GetTime(&time_PS_start);
   double mse_0 = GRU(input_0, y_df_0, 0);
   XTime_GetTime(&time_PS_end);

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
    curr_time = ((float)1.0 * (time_PS_end - time_PS_start) / (5*COUNTS_PER_SECOND));
    printf("\nTime of PS calculation = %f\n", curr_time);
    return 0;
}
double GRU_PL(float input[FRAME_SIZE][VAR_104],float complex y_df[FRAME_SIZE][100][VAR_52], float snr_val) {
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
    float input_data_to_ip[VAR_152], output_data_from_ip[VAR_144];
    int iter;
    double complex temp[VAR_48];
    double temp4[48], temp5[48];
    int status;
    XAxiDma_Config *DMA_confptracp;
    XAxiDma AxiDMAacp;
    DMA_confptracp = XAxiDma_LookupConfig(XPAR_AXI_DMA_0_DEVICE_ID);
    status = XAxiDma_CfgInitialize(&AxiDMAacp, DMA_confptracp);
    if(status != XST_SUCCESS) {
        printf("ACP DMA Init Failed\t\n");
        return XST_FAILURE;
    }
    // Call the functions
    double nmse_sum_frame = 0;
    double mse_sum_iter = 0, nmse_sum=0, hpf=0;
    for(int i = 0; i < FRAME_SIZE; i++){
    	for(int i=0;i<152;i++){
    		input_data_to_ip[i]=0;
    	}
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
			input_data_to_ip[j] = hLS[j];// First we put the hLS as input_0.h data -- 104 Values of hLS
		}
//		for(int j=0;j<VAR_48;j++){
//			input_data_to_ip[j+104] = 0; // Second we put ht_1 a total of 48 values
//		}
		for(int j=0;j<VAR_144;j++){
			output_data_from_ip[j] = 0; // Second we put ht_1 a total of 48 values
		}
//		for(int a =0;a<152;a++){
//			printf("Input Data = %f\n", input_data_to_ip[a]);
//		}


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
//				resetGate(rgiw, hLS, rgib, rghw, ht_1, rghb, r);
//				updateGate(ugiw, hLS, ugib, ughw, ht_1, ughb, z);
//				tanh_layer(tgiw, hLS, tgib, r, tghw, ht_1, tghb, n);
//				ht_new(z, ht_1, n, ht);
//				linear_layer(outWeight, ht, outBias, output_gru[i]);

            	 status = XAxiDma_SimpleTransfer(&AxiDMAacp, (UINTPTR)output_data_from_ip, (sizeof(float )*VAR_144),XAXIDMA_DEVICE_TO_DMA);
            	 status = XAxiDma_SimpleTransfer(&AxiDMAacp, (UINTPTR)input_data_to_ip, (sizeof(float )*VAR_152),XAXIDMA_DMA_TO_DEVICE);
            	 status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR,0x04) & 0x00000002;
            	 while(status!=0x00000002) {
            	       status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR,0x04) & 0x00000002;
            	 }
            	 status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR,0x34) & 0x00000002;
            	 while(status!=0x00000002) {
            	       status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR,0x34) & 0x00000002;
            	 }
 				for(k=0;k<VAR_96;k++){
 					output_gru[i][k]=output_data_from_ip[k];
 				}
				for(k=96;k<VAR_144;k++){
					ht_1[k-96] = output_data_from_ip[k];
				}

				k=0;
	            equalizer(yd_dsym, output_gru[i], i, yeq); // Gives yEqualized output of size 48X1 Complex values

	            demapping(yeq, d, yd, hDPA_complex); // hDPA_complex is now updated


	            for(int j=0;j<VAR_52;j++){
	            	hDPA[j] = (float)crealf(hDPA_complex[j]);
	                hDPA[j+52] = (float)cimagf(hDPA_complex[j]);
	            }

	            for(int j=0;j<VAR_104;j++){
	            	hprev[i][iter][j] = 0.5*hLS[j] + 0.5*hDPA[j];
	            }
            }
            else{
                // Making stream data after every iteration for IP
                for(int w=0;w<VAR_104;w++){
                	input_data_to_ip[w]=hprev[i][iter-1][w];
                }
                for(int w=0;w<VAR_48;w++){
                	input_data_to_ip[w+VAR_104]=ht_1[w];
                }
//				resetGate(rgiw, hprev[i][iter-1], rgib, rghw, ht_1, rghb, r);
//				updateGate(ugiw, hprev[i][iter-1], ugib, ughw, ht_1, ughb, z);
//				tanh_layer(tgiw, hprev[i][iter-1], tgib, r, tghw, ht_1, tghb, n);
//				ht_new(z, ht_1, n, ht);
//				linear_layer(outWeight, ht, outBias, output_gru[i]);
           	 status = XAxiDma_SimpleTransfer(&AxiDMAacp, (UINTPTR)output_data_from_ip, (sizeof(float )*VAR_144),XAXIDMA_DEVICE_TO_DMA);
           	 status = XAxiDma_SimpleTransfer(&AxiDMAacp, (UINTPTR)input_data_to_ip, (sizeof(float )*VAR_152),XAXIDMA_DMA_TO_DEVICE);
           	 status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR,0x04) & 0x00000002;
           	 while(status!=0x00000002) {
           	       status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR,0x04) & 0x00000002;
           	 }
           	 status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR,0x34) & 0x00000002;
           	 while(status!=0x00000002) {
           	       status = XAxiDma_ReadReg(XPAR_AXI_DMA_0_BASEADDR,0x34) & 0x00000002;
           	 }
				for(k=0;k<VAR_96;k++){
					output_gru[i][k]=output_data_from_ip[k];
				}
				for(k=96;k<VAR_144;k++){
					ht_1[k-96] = output_data_from_ip[k];
				}
//				for(k=0;k<VAR_48;k++){
//					ht_1[k] = ht[k];
//				}
				k=0;
	            equalizer(yd_dsym, output_gru[i], i, yeq); // Gives yEqualized output of size 48X1 Complex values - tested and works like a charm

	            demapping(yeq, d, yd, hDPA_complex); // hDPA_complex is now updated


	            for(int j=0;j<VAR_52;j++){
	            	hDPA[j] = (float)crealf(hDPA_complex[j]);
	                hDPA[j+52] = (float)cimagf(hDPA_complex[j]);
	             }
	            // Time Averaging
	            for(int j=0;j<VAR_104;j++){
	            	hprev[i][iter][j] = 0.5*hDPA[j] + 0.5*hprev[i][iter-1][j];
	            }
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
int main_PL()
{

  XTime time_PS_start , time_PS_end;
  float curr_time;
  XTime_SetTime(0);
  XTime_GetTime(&time_PS_start);
   double mse_0 = GRU_PL(input_0, y_df_0, 0);
   XTime_GetTime(&time_PS_end);

       printf("mse_0 PL = %f\n", mse_0);

   double mse_5 = GRU_PL(input_5, y_df_5, 5);
       printf("mse_5 = %f\n", mse_5);

   double mse_10 = GRU_PL(input_10, y_df_10, 10);
       printf("mse_10 PL = %f\n", mse_10);

   double mse_15 = GRU_PL(input_15, y_df_15, 15);
       printf("mse_15 PL= %f\n", mse_15);

   double mse_20 = GRU_PL(input_20, y_df_20, 20);
       printf("mse_20 PL = %f\n", mse_20);

   double mse_25 = GRU_PL(input_25, y_df_25, 25);
       printf("mse_25 PL = %f\n", mse_25);

   double mse_30 = GRU_PL(input_30, y_df_30, 30);
       printf("mse_30 PL = %f\n", mse_30);

   double mse_35 = GRU_PL(input_35, y_df_35, 35);
       printf("mse_35 PL = %f\n", mse_35);

   double mse_40 = GRU_PL(input_40, y_df_40, 40);
//    printf("NMSE:  SNAR_0 = 0.29665, SNR_5 = 0.093882, SNR_10 = 0.02941,SNR_15 = 0.00966, SNR_20 = 0.003298, SNR_25 = 0.001310, SNR_30 = 0.000704, SNR_35 = 0.000496, SNR_40 = 0.000433\n");

    printf("mse_40 = %f", mse_40);
    curr_time = ((float)1.0 * (time_PS_end - time_PS_start) / (5*COUNTS_PER_SECOND));
    printf("\nTime of PL calculation = %f\n", curr_time);
    return 0;
}


int main()
{
    init_platform();

    main_PS();
    main_PL();
    cleanup_platform();
    return 0;
}

