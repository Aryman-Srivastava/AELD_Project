#include <stdio.h>
#include <stdlib.h>
#include "helper.h"
#include "weights.h"
#include <math.h>
#include "input_0.h"

void GRU(hls::stream<axis_data> &hls, hls::stream<axis_data> &output_data);
void GRU_tb(float hls[VAR_104], float ht_1[VAR_48], float output_data[VAR_48]);

int main(){
	float input_data[VAR_104], ht_1[VAR_48];
	float output_data[VAR_48], output_data_tb[VAR_48];
	hls::stream<axis_data> input_stream, output_stream;
	axis_data local_Stream;
	int i;

	for(i=0;i<VAR_104;i++){
		input_data[i] = input_0[0][i];

		local_Stream.data = input_0[0][i];
		local_Stream.last = 0;
		input_stream.write(local_Stream);
	}

	for(i=0;i<VAR_48;i++){
		ht_1[i] = 0;

		local_Stream.data = 0;
		if(i == VAR_48-1){
			local_Stream.last = 1;
		}
		else{
			local_Stream.last = 0;
		}
		input_stream.write(local_Stream);
	}

	GRU_tb(input_data, ht_1, output_data_tb);
	GRU(input_stream, output_stream);

	for(i=0;i<VAR_48;i++){
		local_Stream = output_stream.read();
		output_data[i] = local_Stream.data;
	}

	for(i = 0;i < VAR_48;i++){
		printf("%d\n", abs(output_data[i] - output_data_tb[i]));
	}
}

void GRU_tb(float hls[VAR_104], float ht_1[VAR_48], float output_data[VAR_48]){

	float hls_data[VAR_104], ht[VAR_48];
	float r[VAR_48], z[VAR_48], n[VAR_48];

	resetGate(rgiw, hls, rgib, rghw, ht_1, rghb, r);
	updateGate(ugiw, hls, ugib, ughw, ht_1, ughb, z);
	tanh_layer(tgiw, hls, tgib, r, tghw, ht_1, tghb, n);
	ht_new(z, ht_1, n, output_data);
}
