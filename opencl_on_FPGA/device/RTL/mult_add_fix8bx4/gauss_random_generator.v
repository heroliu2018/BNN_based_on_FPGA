`timescale 1 ps / 1 ps
module gauss_random_generator(
    input   clock,
    input   resetn,
    input   ivalid, 
    input   iready,
    output  ovalid, 
    output  oready,

    input      [7:0]          load,     /*load seed to rand_num,active high */
    input      [31:0]    seed,         
    output reg signed[15:0] rand_output    /*sum */
);
 
assign ovalid = 1'b1;
assign oready = 1'b1;
reg [31:0] rand_num;

reg signed[15:0] rand_temp;

integer i;
reg [15:0] j;
always@(posedge clock or negedge resetn)
begin
    if(!resetn) 
        begin       
            rand_num <= 32'b0;
            rand_temp <= 16'b0;
            rand_output <= 16'b0;
            j <= 16'b0;
        end
    else if(load)begin
            rand_num <=seed;    /*load the initial value when load is active*/
            j <= 16'b0;
            rand_temp <= 16'b0;
        end
    else
        begin             
            for(i=0;i<24;i=i+1)begin
                rand_num[i] <= rand_num[i+1];
            end           
            rand_num[24] <= rand_num[25]^rand_num[0];
            rand_num[25] <= rand_num[26]^rand_num[0];
            rand_num[26] <= rand_num[27];
            rand_num[27] <= rand_num[28];
            rand_num[28] <= rand_num[29];
            rand_num[29] <= rand_num[30]^rand_num[0];
            rand_num[30] <= rand_num[31]; 
            rand_num[31] <= rand_num[0];
            rand_temp  <= rand_temp +  rand_num[0] + rand_num[0] - 1'b1;            
                       
            if(j == 16'h03ff)begin
                rand_output <=rand_temp;
                rand_temp <= 16'b0;                
                j <=16'b0; 				
                end        
			else
				begin
					j <= j+1;
				end
			
        end    	    			          		            
end
endmodule
