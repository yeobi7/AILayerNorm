/*
module Preprocess(
    input						i_clk,
    input            			i_rstn,
    input            			i_valid,
    input 	   	signed	[21:0] 	i_Ex,    
    input      			[31:0] 	i_Ex2,   

	output			 			o_Pre_done,	// Preprocess done
    output 		signed	[21:0] 	o_mean,  
    output reg 			[7:0] 	o_std
);



	////////////// F/F ///////////////
	reg 	 			n_state,	c_state;
    reg signed  [ 8:0]  n_mean, 	c_mean;  // bit extension (signed) 		
//    reg 		[ 2:0]  n_std, 		c_std;
	reg			[43:0]	n_var,		c_var;


    //////////////// reg //////////////// 
    //reg signed	[43:0]  squared_Ex;      // ex^2
    //reg 		[43:0]  var;   			// variance = i_Ex2 - squared_Ex

    
	//////////////// wire //////////////// 
	wire		[15:0]	temp_var;

    // F/F
    always @(posedge i_clk or negedge i_rstn) 
    begin
        if (!i_rstn) 
        begin
            c_state 	<= 1'b0;
            c_mean   	<= 8'sd0;
            //c_std 		<= 8'd0;
            c_var		<= 16'd0;
        end

        else if (i_valid)
        begin
        	c_state		<= n_state;
            c_mean   	<= n_mean;                 
            //c_std	 	<= n_std;
			c_var		<= n_var;
        end
    end


	// n_state
	always @(*)
	begin
		n_state = c_state;
		case (c_state)
		1'b0 : if (i_valid)		 	n_state = 1'b1;
		1'b1 : if (o_Pre_done)		n_state = 1'b0;
		endcase
	end


	// n_mean, n_std
	always @(*)
	begin
		n_mean = c_mean;
		//n_std  = c_std;
		n_var  = c_var;
		case (c_state)
		1'b0 :  begin
					n_mean = 22'sd0;
					//n_std  = 8'd0;
					n_var  = 44'd0;
				end

		1'b1 :  begin
					n_mean = i_Ex;
					//n_std  = c_std;
					n_var  = {12'b0, i_Ex2} - (i_Ex * i_Ex);
				end
		endcase
	end
	

	assign temp_var = c_var[15:0];

	// X^(-0.5) LUT, Q0.8 ex) 0.088 * 256 = 22.5 => 23
	always @(*)
	begin
		casex (temp_var)									// var^(-0.5) -> 1/sigma 		
			16'b1xxx_xxxx_xxxx_xxxx:	o_std = 8'd1;		// 0.0055
			16'b01xx_xxxx_xxxx_xxxx: 	o_std = 8'd2;		// 0.0078
			16'b001x_xxxx_xxxx_xxxx: 	o_std = 8'd3;		// 0.011
			16'b0001_xxxx_xxxx_xxxx: 	o_std = 8'd4;		// 0.0156
			16'b0000_1xxx_xxxx_xxxx: 	o_std = 8'd6;		// 0.022
			16'b0000_01xx_xxxx_xxxx: 	o_std = 8'd8;		// 0.031
			16'b0000_001x_xxxx_xxxx: 	o_std = 8'd11;		// 0.044
			16'b0000_0001_xxxx_xxxx: 	o_std = 8'd16;		// 0.0625
			16'b0000_0000_1xxx_xxxx: 	o_std = 8'd23;		// 0.088
			16'b0000_0000_01xx_xxxx: 	o_std = 8'd32;		// 0.125
			16'b0000_0000_001x_xxxx: 	o_std = 8'd45;		// 0.1768
			16'b0000_0000_0001_xxxx: 	o_std = 8'd64;		// 0.25
			16'b0000_0000_0000_1xxx: 	o_std = 8'd90;		// 0.3535
			16'b0000_0000_0000_01xx: 	o_std = 8'd128;		// 0.5
			16'b0000_0000_0000_001x: 	o_std = 8'd181;		// 0.707
			16'b0000_0000_0000_0001: 	o_std = 8'd255;		// 1
			default: 					o_std = 8'd0; 
        endcase
	end

	
	// ******************** u = 0, sigma = 0 -> ??
	assign o_Pre_done = (o_std) ? 1 : 0;

	assign o_mean = i_Ex;

endmodule
*/


module Preprocess(
    input            			i_valid,
    input 	   	signed	[21:0] 	i_Ex,    
    input      			[31:0] 	i_Ex2,   

	output			 			o_Pre_done,	// Preprocess done
    output 		signed	[21:0] 	o_mean,  
    output reg 			[7:0] 	o_std

);

	reg		[43:0]	var;
	reg		[15:0]	temp_var;
    
	// var
	always @(*)
	begin
		var = 16'd0;
		temp_var = 16'd0;
		if (i_valid)
			begin
				var = {12'b0, i_Ex2} - (i_Ex * i_Ex);
				temp_var = var[15:0];
			end
	end


	// X^(-0.5) LUT, Q0.8 ex) 0.088 * 256 = 22.5 => 23
	always @(*)
	begin
		casex (temp_var)									// var^(-0.5) -> 1/sigma 		
			16'b1xxx_xxxx_xxxx_xxxx:	o_std = 8'd1;		// 0.0055
			16'b01xx_xxxx_xxxx_xxxx: 	o_std = 8'd2;		// 0.0078
			16'b001x_xxxx_xxxx_xxxx: 	o_std = 8'd3;		// 0.011
			16'b0001_xxxx_xxxx_xxxx: 	o_std = 8'd4;		// 0.0156
			16'b0000_1xxx_xxxx_xxxx: 	o_std = 8'd6;		// 0.022
			16'b0000_01xx_xxxx_xxxx: 	o_std = 8'd8;		// 0.031
			16'b0000_001x_xxxx_xxxx: 	o_std = 8'd11;		// 0.044
			16'b0000_0001_xxxx_xxxx: 	o_std = 8'd16;		// 0.0625
			16'b0000_0000_1xxx_xxxx: 	o_std = 8'd23;		// 0.088
			16'b0000_0000_01xx_xxxx: 	o_std = 8'd32;		// 0.125
			16'b0000_0000_001x_xxxx: 	o_std = 8'd45;		// 0.1768
			16'b0000_0000_0001_xxxx: 	o_std = 8'd64;		// 0.25
			16'b0000_0000_0000_1xxx: 	o_std = 8'd90;		// 0.3535
			16'b0000_0000_0000_01xx: 	o_std = 8'd128;		// 0.5
			16'b0000_0000_0000_001x: 	o_std = 8'd181;		// 0.707
			16'b0000_0000_0000_0001: 	o_std = 8'd255;		// 1
			default: 					o_std = 8'd0; 
        endcase
	end


	assign o_mean = i_Ex;
	assign o_Pre_done = (o_std) ? 1 : 0;


endmodule

