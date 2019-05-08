`timescale 1 ps / 1 ps
module gauss_random_generator(
    input   clock,
    input   resetn,
    input   ivalid, 
    input   iready,
    output  ovalid, 
    output  oready,

    input                load,     /*load seed to rand_num,active high */
    input      [15:0]    seed,         
    output reg [7:0]    num_one    /*count the 1's number*/
);
 
assign ovalid = 1'b1;
assign oready = 1'b1;
integer i;
reg [15:0]    rand_num;  /*random number output*/

always@(posedge clock or negedge resetn)
begin
    if(!resetn) 
        begin       
            rand_num <=16'b0;
            num_one  <= 8'b0;
        end
    else if(load)
        rand_num <=seed;    /*load the initial value when load is active*/
    else
        begin
            
            rand_num[0] <= rand_num[15];
            rand_num[1] <= rand_num[0];
            rand_num[2] <= rand_num[1];
            rand_num[3] <= rand_num[2];
            rand_num[4] <= rand_num[3]^~rand_num[15];
            rand_num[5] <= rand_num[4]^~rand_num[15];
            rand_num[6] <= rand_num[5]^~rand_num[15];
            rand_num[7] <= rand_num[6];
            rand_num[8] <= rand_num[7];
            rand_num[9] <= rand_num[8];
            rand_num[10] <= rand_num[9];
            rand_num[11] <= rand_num[10];
            rand_num[12] <= rand_num[11]^~rand_num[15];
            rand_num[13] <= rand_num[12]^~rand_num[15];
            rand_num[14] <= rand_num[13]^~rand_num[15];
            rand_num[15] <= rand_num[14];
            num_one <= 8'b0;
            for(i=0;i<=15;i=i+1)	        
	            num_one <= num_one+rand_num[i];
        end    	    			          
    
		            
end
endmodule
