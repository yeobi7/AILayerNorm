module Stage2 #(
	parameter COUNT = 128,
	parameter CNT_WIDTH = 8
)(
	input          		    i_clk,         
    input           	    i_rstn,      
    input					i_valid,
    input	signed	[ 8:0]	i_x_norm,
    input			[ 7:0]	i_gamma,
    input			[ 7:0]	i_beta,
//    input   signed	[21:0]  i_Ex,
    input   signed	[21:0]  i_mean,
//    input   		[31:0]  i_Ex2,  
    input   		[ 7:0]  i_std,
    input   		[ 1:0]  i_alpha,   	
//    input   		[ 7:0]  i_inv_n,       	// 1/C
    
    output         		   	o_S2_done,      // Stage2 done         
    output  signed	[ 7:0]  o_Norm     
);


	parameter	IDLE = 2'd0;
	parameter	BUSY = 2'd1;
	parameter	DONE = 2'd2;


	/////////////// F/F /////////////////
	reg 		[ 1:0]				n_state,	c_state;
	reg			[CNT_WIDTH-1 :0]	n_cnt,		c_cnt;		
//	reg								n_done,		c_done;
//    reg signed 	[ 7:0]  n_o_Norm,	c_o_Norm;
//	reg signed	[21:0]	n_mean,		c_mean;
//	reg 		[ 7:0]	n_std,		c_std;
	

	///////////////  reg  ////////////////
	//reg	signed	[ 7:0]	temp_norm;
	//reg	signed	[21:0]	temp_mean;
	//reg			[ 7:0]	temp_std;


	/////////////// wire ////////////////
	wire signed	[ 7:0]	temp_norm;
//	wire signed	[21:0]	temp_mean;
//	wire		[ 7:0]	temp_std;

	// F/F
	always @(posedge i_clk or negedge i_rstn) 
    begin
        if (!i_rstn)
        begin
        	c_state	  	<= 1'd0;
            c_cnt	 	<= {CNT_WIDTH{1'b0}};
//            c_done		<= 1'd0;
//            c_o_Norm 	<= 8'sd0;
//            c_mean		<= 8'sd0;
//            c_std		<= 8'd0;
        end

        else 
        begin
        	c_state   	<= n_state;
        	c_cnt		<= n_cnt;
//        	c_done		<= n_done;
//        	c_o_Norm 	<= n_o_Norm;
//        	c_mean		<= n_mean;
//        	c_std		<= n_std;
        end
    end


	// n_state
	always @(*)
	begin
		n_state = c_state;
		case (c_state)
//		IDLE : if (Pre_done)		 	n_state = BUSY;
		IDLE : if (i_valid)				n_state = BUSY;
		BUSY : if (c_cnt % 8 == 7)		n_state = DONE;
		DONE : 							n_state = IDLE;
		endcase
	end

	
	// n_cnt
	always @(*)
	begin
		n_cnt = c_cnt;
		case (c_state)
		IDLE :							n_cnt = 1'd0;
		BUSY :	if (i_valid)			n_cnt = c_cnt + 1;
		endcase
	end

/*
	// n_done
	always @(*)
	begin
		n_done = c_done;
		case (c_state)
		IDLE :							n_done = 1'd0;
		DONE :	if (c_cnt % 8 == 7)		n_done = 1'd1;
		endcase
	end
*/



/*
	// n_o_Norm
	always @(*)
	begin
		n_o_Norm = c_o_Norm;
		case (c_state)
		IDLE : 			 			n_o_Norm = 8'sd0;
		BUSY : if (o_Affine_done)	n_o_Norm = temp_norm;
		endcase
	end
*/


/*	
	// n_mean, n_std
	always @(*)
	begin
		n_mean = c_mean;
		n_std  = c_std;
		case (c_state)
		IDLE :	
				begin
					n_mean = temp_mean;
					n_std  = temp_std;
				end

		DONE : 
				begin
					n_mean = 22'sd0;
					n_std  = 8'd0;
				end
		endcase
	end
*/


	/*	No use - F/F
	// Preprocess
	Preprocess u_Preprocess (
	    .i_clk(i_clk),
        .i_rstn(i_rstn),
        .i_valid(i_valid), 
        .i_Ex(i_Ex),
        .i_Ex2(i_Ex2),
        .o_Pre_done(Pre_done),
        .o_mean(temp_mean),
        .o_std(temp_std)
    );
    */


	/*	Use - Combi    
    // Preprocess
	Preprocess u_Preprocess (
        .i_valid(i_valid), 
        .i_Ex(i_Ex),
        .i_Ex2(i_Ex2),
        .o_Pre_done(Pre_done),
        .o_mean(temp_mean),
        .o_std(temp_std)
    );
	*/

	// Affine Unit
	Affine_Unit u_Affine_Unit (
		.i_clk(i_clk),
        .i_rstn(i_rstn),
//        .i_valid(Pre_done),
        .i_valid(i_valid),
        .i_x_norm(i_x_norm),
        .i_alpha(i_alpha),
//        .i_mean(c_mean),
        .i_mean(i_mean),
//        .i_std(c_std),
        .i_std(i_std),
		.i_gamma(i_gamma),
        .i_beta(i_beta),
        .o_Affine_done(o_Affine_done),
        .o_Norm(temp_norm)
    );


	// output
	assign o_Norm = temp_norm;
	assign o_S2_done = o_Affine_done; 

endmodule


