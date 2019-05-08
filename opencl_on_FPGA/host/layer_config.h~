#include "../device/hw_param.cl"

#define NUM_CONFIG_ITEM  25

// Two layers full connected network 
//[ll][data_n] should ==ll-1][conv_z]
unsigned layer_config[][NUM_CONFIG_ITEM] = {{1,
							28, 28, 1, 28, 28, 1, 400, 400,
							0,
							1, 1, 400, 1, 0, 0, 0,
							0, 4, 4, 400, 0, 1,
							0,
							2},//Layer-1 fc
							{1,
							1, 1, 400, 1, 1, 400, 10, 10,
							2,
							1, 1, 10, 1, 0, 0, 0,
							0, 4, 4, 10, 0, 1,
							0,
							3}//Layer-2 fc
                                                        };

char precision_config[][3] ={{6,  6, 2},//Layer-1  6 =1（original expand） +5(in random number calculation)，data_in =6, 
							 {9, 2, -2},//Layer-2  8 =4（original expand） +5(in random number calculation)						
							};

unsigned input_config[5] = {28, 28, 1, 1}; //original image size(dim1, dim2, dim3), batch size

unsigned output_config[3] = {1, 1, 10};//Layer-8  Note: only one result is extracted and verified

/*
// Alexnet Configuration batch=16
unsigned layer_config[][NUM_CONFIG_ITEM] = {{0,
							227, 227, 3, 11, 11, 3, 96, 96,
							0,
							55, 55, 96, 4, 0, 0, 1,
							1,sudo 27, 96, 3, 2,
							1,
							1},//Layer-1
							{0,
							27, 27, 96, 5, 5, 48, 256, 256,
							0,
							27, 27, 256, 1, 2, 1, 1,
							1, 13, 13, 256, 3, 2,
							1,
							1},//Layer-2
							{0,
							13, 13, 256, 3, 3, 256, 384, 384,
							0,
							13, 13, 384, 1, 1, 0, 1,
							0, 13, 13, 384, 0, 0,
							0,
							1},//Layer-3
							{0,
							13, 13, 384, 3, 3, 192, 384, 384,
							1,
							13, 13, 384, 1, 1, 1, 1,
							0, 13, 13, 384, 0, 0,
							0,
							0},//Layer-4
							{0,
							13, 13, 384, 3, 3, 192, 256, 256,
							0,
							13, 13, 256, 1, 1, 1, 1,
							1, 6, 6, 256, 3, 2,
							0,
							2},//Layer-5  Note: for last conv layer, outputs are write to fc buffer
							{1,
							24, 24, 256, 6, 6, 256, 4096, 4096,  // Note: The input size (dim1/dim2) is the combined data size (batched)
							2,
							4, 4, 4096, 6, 0, 0, 1,
							0, 4, 4, 4096, 0, 0,
							0,
							3},//Layer-6 fc
							{1,
							4, 4, 4096, 1, 1, 4096, 4096, 4096,
							3,
							4, 4, 4096, 1, 0, 0, 1,
							0, 4, 4, 4096, 0, 0,
							0,
							2},//Layer-7 fc
							{1,
							4, 4, 4096, 1, 1, 4096, 1024, 1024,
							2,
							4, 4, 1024, 1, 0, 0, 0,
							0, 4, 4, 1024, 0, 0,
							0,
							3}//Layer-8 fc
							};

char precision_config[][3] ={{8,  0, -4},//Layer-1
							{ 8,  0, -2},//Layer-2
							{ 8,  0, -1},//Layer-3
							{ 8, -1, -1},//Layer-4
							{ 8, -1, -1},//Layer-5
							{11, -1,  0},//Layer-6
							{10,  0,  2},//Layer-7
							{10,  2,  2}//Layer-8
							};

unsigned input_config[5] = {227, 227, 3, 16}; //original image size(dim1, dim2, dim3), batch size

//unsigned output_config[3] = {27, 27, 96};//Layer-1

//unsigned output_config[3] = {6, 6, 256};//Layer-5

//unsigned output_config[3] = {1, 1, 4096};//Layer-6

unsigned output_config[3] = {1, 1, 1024};//Layer-8  Note: only one result is extracted and verified
*/



