/*
 * ------------------------------------------------------
 *
 *      BNN Inference based on FPGA and OpenCL
 *  
 *       Author :Yingh Liu
 *       Data 2019.4.4
 *    contents: the realization for BNN
 *
 * ------------------------------------------------------
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 */

#define USE_ROM 

// The following macros are used for debug
#define DEBUG_MEMRD
#define DEBUG_CONV
//#define DEBUG_POOL
#define DEBUG_MEMWR
//#define DEBUG_LRN
//#define DEBUG_LRN_OUT

#include "hw_param.cl"
#include "rtl_lib.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable

// Define the precision of the data-path
typedef char DPTYPE;
typedef int  MACTYPE;

// Vectorized data type
typedef struct {
   DPTYPE data[VEC_SIZE];
} lane_data;

// Combined vec-data type from multiple lane
typedef struct {
   lane_data lane[LANE_NUM];
} channel_vec;

// Combined scalar data type from multiple lane
typedef struct {
   DPTYPE lane[LANE_NUM];
} channel_scal;

// Vectorized weight, need int size
typedef struct {
   MACTYPE data[VEC_SIZE];
} lane_data_mac;

// Combined vec-data type from multiple lane
typedef struct {
   lane_data_mac lane[LANE_NUM];
} channel_vec_mac;
typedef struct {
   MACTYPE lane[LANE_NUM];
} channel_scal_mac;

channel channel_vec        data_ch    __attribute__((depth(0)));
channel channel_vec_mac    weight_ch  __attribute__((depth(0)));
channel channel_scal_mac   bias_ch    __attribute__((depth(8)));
channel channel_scal       conv_ch    __attribute__((depth(CHN_DEPTH)));
channel channel_scal       pool_ch    __attribute__((depth(CHN_DEPTH)));
channel channel_scal       bypass_ch  __attribute__((depth(CHN_DEPTH)));


// parallel MAC units including (VEC_SIZE-1) multipliers
MACTYPE mac(lane_data input, lane_data_mac weights)
{
	MACTYPE output = MASK_MULT & CZERO;
	
	#pragma unroll
	for(int i=0; i<VEC_SIZE/4; i++){
		output += MASK_MULT & mult_add_fix8bx4(input.data[i*4], weights.data[i*4], input.data[i*4+1], weights.data[i*4+1], input.data[i*4+2], weights.data[i*4+2], input.data[i*4+3], weights.data[i*4+3]);
	}
	return output;
}

DPTYPE pool_max(DPTYPE a_in, DPTYPE b_in)
{
	DPTYPE max_value;
	
	if(a_in >= b_in)
		max_value = a_in;
	else
		max_value = b_in;
	
	return max_value;

}

// Fetch Data from Global Memory
__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void memRead(
			// Params Ports
			uchar  data_dim1,
			uchar  data_dim2,
			ushort data_dim1xdim2,
			uchar  weight_dim1,
			uchar  weight_dim2,
			ushort weight_dim3,
			ushort weight_dim4_div_lane, // avoid generating divider
			uchar  weight_dim1x2,
			uint   weight_dim1x2x3,
			uchar  conv_x,
			//uchar  conv_y,           // not used in this version
			uchar  stride,
			uchar  padding,
			uchar  split,
			uchar  group_num_x,//layer_config[j][conv_x]/CONV_GP_SIZE_X
			uchar  group_num_y,  //layer_config[j][conv_y]/CONV_GP_SIZE_Y
			uchar  group_rem_size_x,// CONV_GP_SIZE_X*layer_config[j][weight_w]
			//uchar  group_rem_size_y, // not used in this version
			uint   group_rem_size_xyz,//rem_x *rem_y *wegiht_n
			uchar  win_size_x,  // weight_w+CONV_GP_SIZE_X-1
			uchar  win_size_y, // it is also weight_h
			uint   win_size_xyz,   
            uint   sample_num,// sampleing numbers
            uchar  sample_control,//control the data sample, because the size of data will expand sample_num times from the second layer
			// Data Ports
			__global lane_data    *restrict bottom,
			__global channel_vec  *restrict weights_mean,
            __global channel_vec  *restrict weights_var,
            __global channel_scal *restrict bias_mean,
			__global channel_scal *restrict bias_var        )

{


	// Input Data, Weights and Bias
	lane_data     data_vec;
	channel_vec   data_ch_vec;
	channel_vec   weight_mean_ch_vec;
    channel_vec   weight_var_ch_vec;
	channel_scal  bias_mean_ch_in;
    channel_scal  bias_var_ch_in;
	ushort        data_offset = 0; // assuming the 1st layer is not in split
	
	// virtual loop counters
	ushort gp_num_x, gp_num_y, out_idx_z; //group counters
	ushort gp_num_x_winbuf, gp_num_y_winbuf, out_idx_z_winbuf;
	uchar  output_idx_dim1, output_idx_dim2;
	ushort output_idx_dim3;
	uchar  win_itm_x, win_itm_y;//loop for transfer data
	ushort win_itm_z;
	
	uchar  gp_item_idx_x;

	ushort feature_idx_dim1, feature_idx_dim2;
	ushort feature_idx_dim3;

	uint   item_loop_bound;	
	uchar  flag; // ping-pong flag
    // for sampling of wieghts and bias
	unsigned char load = 1;
    unsigned short seed_weight = 1024*64-1;//n=16 bit, 2^n-1 states,set the middle
    unsigned short seed_bias = 1024*64-1;

    unsigned short random_weight = 0;
    unsigned char gauss_weight = 0;
    unsigned short random_bias = 0;
    unsigned char gauss_bias =0 ;
    unsigned int  sample_offset =0;// used for  data transfer after the 2 layers ,because it begins sampling
    unsigned char  sample_debug =1;//just for print
    unsigned char  display_debug =0;//just for print
    channel_vec_mac temp_weight;
    channel_scal_mac temp_bias;
	// Ping-pong buffer
	__local lane_data    win_buffer[2][WIN_BUF_SIZE]; // working sequence 0->1->0->1 ...
	// Weight buffer
	__local channel_vec_mac  weight_buffer[WEIGHT_BUF_SIZE];
    
    for(unsigned int sample_i =0; sample_i <sample_num; sample_i++){
	    // Initialize the winbuf with the data in the first iteration of the group looping (as gp_num_x_winbuf=0, gp_num_y_winbuf=0)
	    //extract the data from bottom[]  to data_vec
	    if(sample_i==0)
	        sample_debug=1;
	    else
	        sample_debug=0;
	    for(unsigned short win_itm_z=0; win_itm_z<weight_dim3/VEC_SIZE; win_itm_z++){
		    for(unsigned char  win_itm_y=0; win_itm_y<win_size_y; win_itm_y++){
			    //win_size_x= weight_w+(CONV_GP_SIZE_X-1)*stride
			    for(unsigned char  win_itm_x=0; win_itm_x<win_size_x; win_itm_x++){	
			        feature_idx_dim1 = win_itm_x;
			        feature_idx_dim2 = win_itm_y;
			        feature_idx_dim3 = win_itm_z;
	
			        if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2>=padding && feature_idx_dim2<data_dim2+padding)){
				        //mainly concerns the padding problems
                        if(sample_control==1)
                            sample_offset =sample_i*data_dim1xdim2*weight_dim3;
				        data_vec = bottom[sample_offset + data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + (feature_idx_dim1-padding)];
			        }
			        else{
				        #pragma unroll
				        for(unsigned char vv=0; vv<VEC_SIZE; vv++){
					        data_vec.data[vv] = CZERO;
				        }
			        }
			        //storage a work-group need
			        win_buffer[0][win_itm_z*win_size_y*win_size_x + win_itm_y*win_size_x + win_itm_x] = data_vec;

			    }
		    }
	    }
	    //to here ,have gotten one work-group needed data

	    //if input data is less than GP_size, just need once
	    //otherwise, continue to extract the next work-gropu data 
	    if(group_num_x==1)
		    gp_num_x_winbuf = 0; // there is only one group for FC mode when batch=1
	    else
		    gp_num_x_winbuf = 1; // loop start from the second group
	    gp_num_y_winbuf = 0;
	    out_idx_z_winbuf = 0;
	
	    // reset global group virtual loop counters
	    gp_num_x = 0;
	    gp_num_y = 0;
	    out_idx_z = 0;
    
	
        Group:for(unsigned int out_idx_xyz=0; out_idx_xyz<(weight_dim4_div_lane*group_num_y*group_num_x); out_idx_xyz++){

	        // special case when split==1, the output feature maps depend on only half the input feature maps
	        if(split==0)
		        data_offset = 0;
	        else if(out_idx_z_winbuf<(weight_dim4_div_lane>>1)) // the lower half of the output feature maps depend on the lower half of the input
		        data_offset = 0;
	        else
		        data_offset = weight_dim3/VEC_SIZE;	// the upper half of the output feature maps depend on the upper half of the input

	        flag = out_idx_xyz & 0x01; //ping-pong flag
	
	        // reset output loop counters         for weights
	        output_idx_dim1 = 0;
	        output_idx_dim2 = 0;
	        output_idx_dim3 = 0;
	        // reset in-group (item counters )
	        gp_item_idx_x = 0;
	
	        // reset input winbuffer loop counters   for data
	        win_itm_x = 0;
	        win_itm_y = 0;
	        win_itm_z = 0;
	
	        //calculate the numbers of items
	        if(gp_num_x==group_num_x-1)//win_size_x :one group's size;group_rem_size_x:size for no repeated data
		        //win_size_xyz/VEC_SIZE: the numbers of all work-groups  the max
		        item_loop_bound = win_size_x>=group_rem_size_x?(win_size_xyz/VEC_SIZE):(group_rem_size_xyz/VEC_SIZE);
	        else{
		        item_loop_bound = (weight_dim1x2x3*CONV_GP_SIZE_Y*CONV_GP_SIZE_X/VEC_SIZE);//include vec 's number
	        }
	        //ignore assumed data dependences that inhibit vectorization
	        #pragma ivdep array(win_buffer)
	        #pragma ivdep array(weight_buffer)
	        Item:for(unsigned int  win_itm_xyz=0; win_itm_xyz<item_loop_bound; win_itm_xyz++){       
		        // Winbuffer loading operations
		        if(win_itm_z<weight_dim3/VEC_SIZE){
			        //these three parameters control the item process in a work-group
			        feature_idx_dim1 = win_itm_x+gp_num_x_winbuf*CONV_GP_SIZE_X*stride;
			        feature_idx_dim2 = win_itm_y+gp_num_y_winbuf*CONV_GP_SIZE_Y*stride;
			        feature_idx_dim3 = win_itm_z;

			        if((feature_idx_dim1>=padding && feature_idx_dim1<data_dim1+padding) && (feature_idx_dim2>=padding && feature_idx_dim2<data_dim2+padding)){
			            if(sample_control==1)
                            sample_offset =sample_i*data_dim1xdim2*weight_dim3;
				        data_vec = bottom[sample_offset + data_offset*data_dim1xdim2 + feature_idx_dim3*data_dim1xdim2 + (feature_idx_dim2-padding)*data_dim1 + (feature_idx_dim1-padding)];
			        }
			        else{ // for padding (feature_idx<padding or data_dim+padding<=feature_idx<data_dim+2*padding)
				        #pragma unroll
					        for(unsigned char vv=0; vv<VEC_SIZE; vv++){
						        data_vec.data[vv] = CZERO;
				                }
	                   }
			
			        win_buffer[(~flag)&0x01][win_itm_z*win_size_y*win_size_x + win_itm_y*win_size_x + win_itm_x] = data_vec;

			        // used as loop counters
			        if((win_itm_z==weight_dim3/VEC_SIZE-1) && (win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1))
				        win_itm_z = 0;
			        else if((win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1))
				        win_itm_z++;

			        if((win_itm_y==win_size_y-1) && (win_itm_x==win_size_x-1))
				        win_itm_y = 0;
			        else if(win_itm_x==win_size_x-1)
				        win_itm_y++;
			
			        if(win_itm_x==win_size_x-1)
				        win_itm_x = 0;
			        else
				        win_itm_x++;
				
		        }
		
		        // Load weight into weight buffer
		        if(gp_item_idx_x==0){
			        //output_idx_dim3  =weight_n/Vec_size
			        weight_mean_ch_vec = weights_mean[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
                    weight_var_ch_vec = weights_var[out_idx_z*weight_dim1x2x3/VEC_SIZE + output_idx_dim3*weight_dim1x2 + output_idx_dim2*weight_dim1 + output_idx_dim1];
                    
                    load = 1;
                    seed_weight = seed_weight >>1;//n=16 bit, 2^n-1 states,set the middle    
                    if(seed_weight < 2)
                            seed_weight = 1024*64-1;
                    random_weight = gauss_random_generator(load, seed_weight);
                    //printf("weight_random_seed: %d \n", random_weight);
                    load = 0;
                    #pragma unroll      
                    for(unsigned char vec_i =0; vec_i <LANE_NUM; vec_i++){
                            for(unsigned char vec_j =0; vec_j <VEC_SIZE; vec_j++){
                                    if(sample_debug){
                                        temp_weight.lane[vec_i].data[vec_j]=16*weight_mean_ch_vec.lane[vec_i].data[vec_j];
                                        #ifdef DEBUG_MEMRD
                                        if((win_itm_xyz==0)&&(vec_j==0))
                                            printf("weight_mean= %d  ", temp_weight.lane[vec_i].data[vec_j]);
                                        #endif
                                    }
                                    else
                                    {  
                                        random_weight = gauss_random_generator(load, seed_weight);  
                                        for(unsigned int gauss_i = 0;gauss_i<16;gauss_i++){
                                            gauss_weight =gauss_weight + (random_weight & 0x0001);
                                            random_weight = random_weight>>1;
                                        }
                                        //if(sample_debug < SAMPLE_NUM_DEBUG){
                                        //    printf("weight_random_number: %d ---", gauss_weight); 
                                       //     sample_debug +=1;   
                                        //}  
                                        temp_weight.lane[vec_i].data[vec_j] = 16*weight_mean_ch_vec.lane[vec_i].data[vec_j] + (4*(gauss_weight-8))*weight_var_ch_vec.lane[vec_i].data[vec_j];//all multify 16
                                        gauss_weight =0;
                                        random_weight=0;
                                    }
                            }
                    }
                    weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1] = temp_weight;                                       
		        }
		        // In this version, grouping is only performed in row (x) direction
		        if(gp_num_x*CONV_GP_SIZE_X+gp_item_idx_x<conv_x){

			        if(output_idx_dim1==0 && output_idx_dim2==0 && output_idx_dim3==0){
				        bias_mean_ch_in = bias_mean[out_idx_z];
                        bias_var_ch_in = bias_var[out_idx_z];
                        load = 1;
                        seed_bias = seed_bias >>1;//n=16 bit, 2^n-1 states,set the middle    
                        if(seed_bias < 2)
                                seed_bias = 1024*64-1;                                                                  
                        random_bias = gauss_random_generator(load, seed_bias);
                        //printf("bias_random_seed: %d", random_bias);
                        load =0;
                        #pragma unroll      
                        for(unsigned char vec_i =0; vec_i <LANE_NUM; vec_i++){
                            if(sample_debug){
                                temp_bias.lane[vec_i] = 16*bias_mean_ch_in.lane[vec_i];
                                #ifdef DEBUG_MEMRD        
                                if((win_itm_xyz==0))
                                    printf("bias_mean= %d  ", temp_bias.lane[vec_i]);
                                #endif
                            }    
                            else{   
                                random_bias = gauss_random_generator(load, seed_bias);  
                                for(unsigned int gauss_i=0;gauss_i<16;gauss_i++){
                                        gauss_bias =gauss_bias + (random_bias & 0x0001);
                                        random_bias = random_bias>>1;
                                }      
                                temp_bias.lane[vec_i] = 16*bias_mean_ch_in.lane[vec_i] + (4*(gauss_bias-8))*bias_var_ch_in.lane[vec_i];//all multify 16
                                gauss_bias =0;
                                random_bias =0;
                            }
                        }
                        write_channel_intel(bias_ch, temp_bias);
				        //#ifdef DEBUG_MEMRD
				        //printf("work-item x=%d, y=%d, z=%d, channel =0, write bias=%f\n", output_idx_dim1, output_idx_dim2, output_idx_dim3, bias_ch_in.lane[0]);
				        //#endif
			        }

			        data_vec = win_buffer[flag][output_idx_dim3*win_size_y*win_size_x + output_idx_dim2*win_size_x + (output_idx_dim1+gp_item_idx_x*stride)];
			        //CU parallel
			        #pragma unroll
			        for(unsigned char ll=0; ll<LANE_NUM; ll++){
				        data_ch_vec.lane[ll] = data_vec;
			        }
			        write_channel_intel(data_ch, data_ch_vec);	
			        
			
			        // weight and bias fetcher,  use weight_mean_ch_vec represents weight_ch_vec
			        temp_weight = weight_buffer[output_idx_dim3*weight_dim2*weight_dim1 + output_idx_dim2*weight_dim1 + output_idx_dim1];
			        write_channel_intel(weight_ch, temp_weight);	
			
			        #ifdef DEBUG_MEMRD
			        if(gp_num_x==group_num_x-1 && gp_num_y==0 && out_idx_z==0){
                        display_debug++;
                        if((display_debug%10)==0){
                            display_debug =0;
				            printf("work-item x=%d, y=%d, z=%d, in channel: data_ch=%f, weight=%f,bias=%f---", output_idx_dim1, output_idx_dim2, output_idx_dim3, (float)data_ch_vec.lane[0].data[0],(float)temp_weight.lane[0].data[0],(float)temp_bias.lane[0]);
                        }			        
                    }
			        #endif

			        // used as output loop counters   output_idx_dim3++
			        if((output_idx_dim3==weight_dim3/VEC_SIZE-1) && (output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1)){
				        output_idx_dim3 = 0;
				        gp_item_idx_x++;
			        }
			        else if((output_idx_dim2==weight_dim2-1)&& (output_idx_dim1==weight_dim1-1))
				        output_idx_dim3++;
			
			        if((output_idx_dim2==weight_dim2-1) && (output_idx_dim1==weight_dim1-1))
				        output_idx_dim2 = 0;
			        else if(output_idx_dim1==weight_dim1-1)
				        output_idx_dim2++;

			        if(output_idx_dim1==weight_dim1-1)
				        output_idx_dim1 = 0;
			        else
				        output_idx_dim1++;

		        }//if(gp_num_x*CONV_GP_SIZE_X+gp_item_idx_x<conv_x)

            }//item

	        // used as virtual group loop counters for winbuf loading operationsžüÐÂ ÏÂÒ»žögroupµÄdataÊýŸÝ
	        if((out_idx_z_winbuf==weight_dim4_div_lane-1) && (gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
		        out_idx_z_winbuf = 0;
	        else if((gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
		        out_idx_z_winbuf++;	

	        if((gp_num_y_winbuf==group_num_y-1) && (gp_num_x_winbuf==group_num_x-1))
		        gp_num_y_winbuf = 0;
	        else if(gp_num_x_winbuf==group_num_x-1)
		        gp_num_y_winbuf++;	

	        if(gp_num_x_winbuf==group_num_x-1)
		        gp_num_x_winbuf = 0;
	        else
		        gp_num_x_winbuf++;
	
	        // used as virtual group loop counters
	        if((out_idx_z==weight_dim4_div_lane-1) && (gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
		        out_idx_z = 0;
	        else if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
		        out_idx_z++;	
            
	        if((gp_num_y==group_num_y-1) && (gp_num_x==group_num_x-1))
		        gp_num_y = 0;
	        else if(gp_num_x==group_num_x-1)
		        gp_num_y++;
            
	        if(gp_num_x==group_num_x-1)
		        gp_num_x = 0;
	        else
		        gp_num_x++;

        }//for group
        printf("\nMem RD sampling once completed\n");
	}//for sample number
	printf("MemRD Kernel 0 lanched !!!\n");
}


__kernel
__attribute__((task))
__attribute__((max_global_work_dim(0)))
void coreConv(
			// Params Ports
			uint  output_num,    //conv_x*y*weight_m/LANE_NUM
			uint  conv_loop_cnt,//weight_w*h*n/VEC_SIZE
			uint  control, //[0]-> relu  [1]->bypass pooling
            uint  sample_num,// sampleing numbers
			char  frac_w,
			char  frac_din,
			char  frac_dout
			)
{
	channel_vec        mac_data;
 	channel_vec_mac    mac_weight;
	channel_scal_mac   bias_ch_out;
	channel_scal       conv_ch_in;
	MACTYPE bias[LANE_NUM];
	MACTYPE conv_out[LANE_NUM];
	MACTYPE lane_accum[LANE_NUM];
	MACTYPE accum_piped[LANE_NUM][PIPE_DEPTH];
	MACTYPE conv_sign_exten[LANE_NUM];
	MACTYPE conv_with_rnd_bit[LANE_NUM];
	MACTYPE conv_sum_bias[LANE_NUM];
	DPTYPE  conv_final[LANE_NUM];
        
        
    //weight and bias should divde 16---fix point
	// each iteration generates one output
    for(unsigned int sample_i =0; sample_i <sample_num; sample_i++){
        for(unsigned int k=0; k<output_num; k++){	

	        bias_ch_out = read_channel_intel(bias_ch);                
	        #pragma unroll
	        for(unsigned char ll=0; ll<LANE_NUM; ll++){

		        conv_out[ll] = CZERO;
		        bias[ll] = bias_ch_out.lane[ll];

		        #pragma unroll
		        for(unsigned int p=0; p<PIPE_DEPTH; p++){//used for saving the results
			        accum_piped[ll][p] = MASK_ACCUM & CZERO;
		        }
	        }

	        for(int j=0; j<conv_loop_cnt; j++){

		        mac_data = read_channel_intel(data_ch);
		        mac_weight = read_channel_intel(weight_ch);

		        #pragma unroll
		        for(unsigned char ll=0; ll<LANE_NUM; ll++){
			
			        lane_accum[ll] = (MASK_ACCUM & accum_piped[ll][PIPE_DEPTH-1]) + (MASK_MULT & mac(mac_data.lane[ll], mac_weight.lane[ll]));
		
			        #pragma unroll
			        for(unsigned int p=PIPE_DEPTH-1; p>0; p-- ){
				        accum_piped[ll][p] = MASK_ACCUM & accum_piped[ll][p-1];
			        }
			
			        accum_piped[ll][0] = MASK_ACCUM & lane_accum[ll];

			        #ifdef DEBUG_CONV
			        if(ll==0 && k==0){
			        	printf("Conv orignal:dot_cnt=%d data=%f weight=%f (loop=%d, lane= %d, vec=0)\n", k, (float)mac_data.lane[ll].data[0], (float)mac_weight.lane[ll].data[0], j, ll);
			        }
			        #endif
		        }
	        }// end of conv loop

	        #pragma unroll
	        for(unsigned char ll=0; ll<LANE_NUM; ll++){

		        #pragma unroll
		        for(unsigned i=0; i<PIPE_DEPTH; i++){
			        conv_out[ll] += accum_piped[ll][i];
		        }
		        //conv_out[ll] += bias[ll];
		        if(conv_out[ll]>=0)
			        conv_sign_exten[ll] = 0x00;
		        else
			        conv_sign_exten[ll] = ~(0xFFFFFFFF>>(frac_w+frac_din-frac_dout-1));//{4,  0, -2}		
		        conv_with_rnd_bit[ll] = (conv_sign_exten[ll] | (conv_out[ll]>>(frac_w+frac_din-frac_dout-1))) + 0x01;
		        
		        if(bias[ll]>=0)
			        conv_sign_exten[ll] = 0x00;
		        else
			        conv_sign_exten[ll] = ~(0xFFFFFFFF>>(frac_w-frac_dout-1));//{4,  0, -2}
                conv_sum_bias[ll] = (conv_sign_exten[ll] |(bias[ll]>>(frac_w-frac_dout-1)))+ 0x01;//bias need to handle 
                conv_with_rnd_bit[ll] += conv_sum_bias[ll];
                
		        if(conv_with_rnd_bit[ll]>=256)
			        conv_with_rnd_bit[ll] = MASK9B & 0xFF;
		        else if(conv_with_rnd_bit[ll]<-256)
			        conv_with_rnd_bit[ll] = MASK9B & 0x100;		  			        
		        conv_final[ll] = MASK8B & (conv_with_rnd_bit[ll]>>0x01);
		
		        // Relu operation
		        if((control&0x01)==0x01){
			        if((conv_final[ll]&MASKSIGN)==MASKSIGN)
				        conv_ch_in.lane[ll] = 0;
			        else
				        conv_ch_in.lane[ll] = conv_final[ll];
		        }
		        else
			        conv_ch_in.lane[ll] = conv_final[ll];
		
		        #ifdef DEBUG_CONV
		        if(k==0)
			        printf("Conv result:dot_cnt=%d conv_mul=%f bias=%f sum=%f final=%f conv_relu =%f\n", k, (float)conv_out[ll], (float)conv_sum_bias[ll],(float)conv_with_rnd_bit[ll],(float)conv_final[ll], (float)conv_ch_in.lane[ll]);
		        #endif

	        }

	        // write convoluation results
	        if((control&0x02)==0x02)
		        //by-pass pooling
		        write_channel_intel(bypass_ch, conv_ch_in);
	        else // to pooling kernel
		        write_channel_intel(conv_ch, conv_ch_in);
		        //printf("Write channel item-%d is written in channel %d...\n", k, ll);

        }// end of output loop
        printf("Conv sampling once completed\n");
    }//end of sample number
    printf("Conv Kernel 0 lanched !!!\n");
}


__kernel
__attribute__((task))// not modify-2019.4.8
void maxPool(
			// Params Ports
			uint  input_num,
			uchar line_size,  // line_size should be no larger than POOL_LBUF_DEPTH
			uchar pool_size,  // by now, only pooling size no larger than 3
			uchar pool_stride
			
			)
{
	channel_scal conv_ch_out;
	channel_scal pool_final;

	DPTYPE line_buf_0[LANE_NUM][POOL_LBUF_DEPTH];
	DPTYPE line_buf_1[LANE_NUM][POOL_LBUF_DEPTH];
	uchar  line_buf_ptr;
	uchar  col_pool_cnt;
	uchar  row_pool_cnt;
	uchar  row_cnt;
	DPTYPE row_pool_reg[LANE_NUM];
	DPTYPE col_pool_reg[LANE_NUM];
	DPTYPE pool_reg[LANE_NUM][POOL_MAX_SIZE];
	
	// Each iteration consumes one output from convolution kernel
	// and then Pooling is performed in column and row directions
	line_buf_ptr = 0;
	row_pool_cnt = 0;
	col_pool_cnt = 0;
	for(unsigned int k=0; k<input_num; k++){

		conv_ch_out = read_channel_intel(conv_ch);
	
		// Two line buffer to form the 3x3 pooling window
		#pragma unroll
		for(unsigned char ll=0; ll<LANE_NUM; ll++){

			if(pool_size==3)
				row_pool_reg[ll] = pool_max(line_buf_1[ll][line_buf_ptr], line_buf_0[ll][line_buf_ptr]);
			else // pool_size==2
				row_pool_reg[ll] = line_buf_0[ll][line_buf_ptr];
			
			pool_reg[ll][0] = pool_max(row_pool_reg[ll], conv_ch_out.lane[ll]);
			
			if(pool_size==3)
				col_pool_reg[ll] = pool_max(pool_reg[ll][1], pool_reg[ll][2]);
			else //pool_size==2
				col_pool_reg[ll] = pool_reg[ll][1];

			pool_final.lane[ll] = pool_max(col_pool_reg[ll], pool_reg[ll][0]);

			line_buf_1[ll][line_buf_ptr] = line_buf_0[ll][line_buf_ptr];
			line_buf_0[ll][line_buf_ptr] = conv_ch_out.lane[ll];

			#pragma unroll
			for(unsigned char p=POOL_MAX_SIZE-1; p>0; p--){
				pool_reg[ll][p]=pool_reg[ll][p-1];
			}
		}
		
		#ifdef DEBUG_POOL
		printf("Maxpool input_num=%d, line_buf_ptr=%d, row_pool_cnt=%d, col_pool_cnt=%d\n", k, line_buf_ptr, row_pool_cnt, col_pool_cnt);
		printf("        row_cnt=%d\n", row_cnt);
		#endif
		
		// Generates pooling pipeline register wr/rd pointer
		if(row_pool_cnt==(pool_size-1)){

			if(col_pool_cnt==(pool_size-1)){
				write_channel_intel(pool_ch, pool_final);
				#ifdef DEBUG_POOL
				printf("        reg0=%f, reg1=%f, reg2=%f, max=%f\n", (float)pool_reg[0][0], (float)pool_reg[0][1], (float)pool_reg[0][2], (float)pool_final.lane[0]);
				#endif

				col_pool_cnt = (pool_size-pool_stride);
			}
			else
				col_pool_cnt = col_pool_cnt + 1;
		}
		else
			col_pool_cnt = 0;

		// Generates line buffer wr/rd pointer
		if(line_buf_ptr==(line_size-1)){
			line_buf_ptr = 0;

			// Row counters for recognize frames
			if(row_cnt == (line_size-1)) // assuming row_num = line_size, i.e. rectangular frame
				row_cnt = 0;
			else
				row_cnt = row_cnt + 1;

			// Pooling window slide counter for rows
			if(row_cnt == 0)
				row_pool_cnt = 0;
			else if(row_pool_cnt==(pool_size-1))
				row_pool_cnt = (pool_size-pool_stride);
			else
				row_pool_cnt = row_pool_cnt + 1;
		}
		else{
			line_buf_ptr = line_buf_ptr + 1;
		}

	}
}


// Store Data to Global Memory
__kernel
__attribute__((reqd_work_group_size(1,1,LANE_NUM)))
void memWrite(
				// Params Ports
				uchar  out_dim1,// conv_x
				uchar  out_dim2,// conv_y
				ushort out_dim3,// conv_z
				ushort out_dim1xbatch, // out_dim1 x1   sqrt(batch_size)
				uint   out_dim1x2xbatch, // out_dim1 x out_dim2 x batch_size*batch_size
				uchar  batch_indx_dim1,//0
				uchar  batch_indx_dim2,//0
				uchar  bypass,
				uchar  padd_offset,
                uint   sample_num,// sampleing numbers
				// Data Ports
                __global DPTYPE *restrict top
				)
{
	uchar  global_x = get_global_id(0); //=conv_x*sample_n     max value 256
	uchar  global_y = get_global_id(1); //=conv_y max value 256
	ushort global_z = get_global_id(2); //=layer_config[j][weight_m]*sample_num
	uchar  local_x 	= get_local_id(0); //=1 max value 256
	uchar  local_y 	= get_local_id(1); //=1 max value 256
	uchar  local_z 	= get_local_id(2); //=LANE_NUM max value 256

	uchar  index_z_item; // max value 256
	ushort index_z_group;// max value 4096
    //ushort sample_i = 0;
	channel_scal   output;
	__local DPTYPE buffer[LANE_NUM];
	if(local_z==0){
		if((bypass&0x01)==0x01)
			output = read_channel_intel(bypass_ch);
		else
			output = read_channel_intel(pool_ch);

		#pragma unroll
		for(uchar ll=0; ll<LANE_NUM; ll++){
			buffer[ll]=output.lane[ll];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);


	// fetch data from local buffer and write back to DDR
	index_z_group = (global_z-padd_offset)/VEC_SIZE;
	index_z_item  = (global_z-padd_offset)%VEC_SIZE;

	if((global_z-padd_offset)<(out_dim3*sample_num) && (global_z>=padd_offset)){              
        //sample_i = global_z/out_dim3;
		top[index_z_group*out_dim1x2xbatch*VEC_SIZE + (global_y+batch_indx_dim2*out_dim2)*out_dim1xbatch*VEC_SIZE + (global_x+batch_indx_dim1*out_dim1)*VEC_SIZE + index_z_item] = buffer[local_z];

		#ifdef DEBUG_MEMWR
		//if((global_z-padd_offset) == 0){
			//for(unsigned char ll=0; ll<LANE_NUM; ll++){
		printf("MemWr results= %f top=%f (x=%d, y=%d, z=%d, ll=%d)----", (float)output.lane[0],(float)buffer[local_z], global_x, global_y, global_z, 0);
			//}
		//	}
		#endif
	}	
	barrier(CLK_LOCAL_MEM_FENCE);
    //printf("MemWR Kernel 0 lanched !!!----");
}

/*
__kernel
__attribute__((max_work_group_size(LRN_MAX_LOCAL_SIZE)))
void lrn(
			// Params Ports
			uchar data_dim1,
			uchar data_dim2,
			char  frac_dout,
			// Data Ports
			__global lane_data *restrict bottom,
			__global lane_data *restrict top
		)
{
	uchar  global_x = get_global_id(0); // max value 256
	uchar  global_y = get_global_id(1); // max value 256
	ushort global_z = get_global_id(2); // max value 4096

	#ifdef DEBUG_LRN
	int local_x = get_local_id(0);
	int local_y = get_local_id(1);
	int local_z = get_local_id(2);
	int block_x = get_group_id(0);
	int block_y = get_group_id(1);
	int block_z = get_group_id(2);
	#endif
	
	__local DPTYPE z_buffer[VEC_SIZE*LRN_MAX_LOCAL_SIZE+LRN_WIN_SIZE]; // allocate two more points for padding
	__local DPTYPE lrn_buffer[VEC_SIZE*LRN_MAX_LOCAL_SIZE];
	channel_scal data_in;
	channel_scal data_pad_left;
	channel_scal data_pad_right;
	channel_scal data_out;
	lane_data    data_in_partial;
	lane_data    data_left_partial;
	lane_data    data_right_partial;
	lane_data    data_out_partial;
	int          *convert_ptr;
	int          expo;
	uint         manti;
	uint         addr_1, addr_2, addr;
	float        lrn_reg1, lrn_reg2, lrn_tmp, lrn_out;
	short        lrn_cnvt, lrn_cnvt2;
	
	// Load the all data in one line along dim3 into local line buffer
	#pragma unroll
	for(unsigned char ll=0; ll<VEC_SIZE; ll++){
		z_buffer[global_z*VEC_SIZE+ll+LRN_WIN_SIZE/2] = bottom[global_z*data_dim2*data_dim1 + global_y*data_dim1+ global_x].data[ll];
	}
	
	//Padding left
	if(global_z==0){
		#pragma unroll
		for(unsigned char ll=0; ll<LRN_WIN_SIZE/2; ll++){
			z_buffer[ll] = CZERO;
		}
	}

	// Padding right
	if(global_z==(get_global_size(2)-1)){
		#pragma unroll
		for(unsigned char ll=0; ll<LRN_WIN_SIZE/2; ll++){
			z_buffer[VEC_SIZE*get_local_size(2)+ll+LRN_WIN_SIZE/2] = CZERO;
		}
	}

	#ifdef DEBUG_LRN
	if(global_z==0&&global_x==0&&global_y==0)
	printf("Kernel LRN: work-item x=%d, y=%d, z=%d(z_local=%d)\n", global_x, global_y, global_z, local_z);
	#endif
	barrier(CLK_LOCAL_MEM_FENCE);

	// Piecewise interpolation pipeline for lrn operation (i.e., y=pwlf(x'))
	for(unsigned char ll=0; ll<VEC_SIZE; ll++){
		// First Step: Coefficients table looking-up
		// Calculate x'=sum(x(k)^2) for the pwlf function, x(k)s are from adjacent featuremaps
		lrn_reg2 = CZERO;
		#pragma unroll
		for(char k=-LRN_WIN_SIZE/2; k<=LRN_WIN_SIZE/2; k++){
			lrn_cnvt = z_buffer[global_z*VEC_SIZE+ll+k+LRN_WIN_SIZE/2]<<(-frac_dout);
			lrn_reg1 = convert_float(lrn_cnvt);
			lrn_reg2 += lrn_reg1 * lrn_reg1;
			#ifdef DEBUG_LRN
			if(global_z==0&&global_x==0&&global_y==0)
			printf("x=%f(k=%d), ", lrn_reg1, k);
			#endif
		}
		convert_ptr = (int*) (&lrn_reg2);
		expo = (EXP_MASK & (*convert_ptr >> MAN_BITS)) - 127;
		manti = ((*convert_ptr) & MAN_MASK);
		
		addr_1 = ((expo-EXP_STEP_MIN)>>EXP_STEP_LOG)<<MAN_INDEX_BITS;
		addr_2 = (manti>>(MAN_BITS-MAN_INDEX_BITS) & MAN_INDEX_MASK)+1;
		if(expo<EXP_STEP_MIN)
			addr = 0;
		else
			addr = addr_1+addr_2;

		lrn_tmp = ((lrn_reg2-x_sample[addr])*h_inv[addr])*coef1[addr] + coef0[addr];	
		
		lrn_cnvt2 = z_buffer[global_z*VEC_SIZE+ll+LRN_WIN_SIZE/2]<<(-frac_dout);
		lrn_out = lrn_tmp*convert_float(lrn_cnvt2);

		// Convert float to DPTYPE fixed-point
		// Note: current version only support frac_din=0 for next layer
		lrn_buffer[global_z*VEC_SIZE+ll] = convert_char_rte(lrn_out);

		#ifdef DEBUG_LRN
		if(global_z==0&&global_x==0&&global_y==0)
		printf("\nKernel LRN (ll=%d): pwlf_x=%f, expo=%d, addr=%d, pwlf_y=%f, lrn=%f\n", ll, lrn_reg2, expo, addr, lrn_tmp, lrn_out);
		#endif
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the results back to global mem
	#pragma unroll
	for(unsigned char vv=0; vv<VEC_SIZE; vv++){
		data_out_partial.data[vv]=lrn_buffer[global_z*VEC_SIZE+vv];
	}
	top[global_z*data_dim2*data_dim1 + global_y*data_dim1 + global_x] = data_out_partial;
	
	#ifdef DEBUG_LRN_OUT
	if(global_z==0&&global_x==0&&global_y==0)
	printf("\nKernel LRN OUT: x=%d, y=%d, z=%d, result=%f\n", global_x, global_y, global_z, (float)data_out_partial.data[0]);
	#endif

}
*/

