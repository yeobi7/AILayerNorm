module TOP_AILayerNorm #(
	parameter	DATA_WIDTH = 192
)(
	input								i_clk,
	input								i_rstn,
	input								i_start,

	input			[DATA_WIDTH-1:0] 	i_x,		// (8bits * 8 input) * N, batch = N

	input		 	[1:0]				i_alpha,
	input 			[7:0]				i_inv_n,
//	input signed	[7:0] 				i_beta,
//	input signed	[7:0]				i_gamma,

	input signed	[DATA_WIDTH-1:0] 	i_beta,
	input signed	[DATA_WIDTH-1:0]	i_gamma,


	output 								o_done,		// AILayerNorm done
	output signed 	[7:0]				o_AILayerNorm
);


	localparam	COUNT	  	= DATA_WIDTH / 8;
	localparam  CNT_WIDTH 	= $clog2(COUNT);

	localparam	IDLE 		= 2'd0;
	localparam	RUN	 		= 2'd1;
	localparam 	DONE 		= 2'd2;




	//////////////////// FlipFlop ////////////////////
	reg			[ 1:0]			c_state,	n_state;
	reg			[CNT_WIDTH:0]	c_cnt,		n_cnt;			// 0 ~ 39 count
	reg			[CNT_WIDTH:0]	c_cnt_o,	n_cnt_o;
	reg			[ 7:0]			c_S1_i, 	n_S1_i;

	/////////////// ************* ////////////////////
//	reg	signed	[ 7:0]			c_gamma, 	n_gamma;
//	reg	signed	[ 7:0]			c_beta, 	n_beta;

	reg	signed	[21:0]			c_S2_mean, 	n_S2_mean;
	reg			[ 7:0]			c_S2_std, 	n_S2_std;
	reg	signed	[21:0]			c_Ex_o,  	n_Ex_o;
	reg			[31:0]			c_Ex2_o,	n_Ex2_o;
//	reg 		[ 8:0]			c_buf_i,	n_buf_i;
	reg			[ 7:0]			c_S2_o,		n_S2_o;

	reg							c_en_S1,	n_en_S1;
//	reg							c_en_S2, 	n_en_S2;
	reg							c_en_wr,	n_en_wr;
	reg							c_en_rd,	n_en_rd;
		




	//////////////////// wire ////////////////////
//	wire 		[ 7:0]	S1_in;
	wire				S1_done;
	wire				S2_done;


	wire signed	[21:0]	Ex_out;
	wire		[31:0]	Ex2_out;


	wire		[ 8:0] 	buffer_in;
	wire		[ 8:0] 	buffer_out;
	wire				buffer_full;
	wire				buffer_empty;

	
	wire signed	[21:0]	Pre_mean;
	wire		[ 7:0]	Pre_std;
	wire				Pre_done;

	wire signed	[ 7:0]  gamma_i;
	wire signed	[ 7:0]  beta_i;		

	wire 		[ 7:0] 	S2_out;
	
	

	/////////////////////////////////////////////////////////////
	// State Transition
	always @(posedge i_clk, negedge i_rstn)
	begin
		if(!i_rstn)
		begin
			c_state		<= 2'd0;	
    		c_cnt		<= {CNT_WIDTH{1'b0}};
     		c_cnt_o		<= {CNT_WIDTH{1'b0}};  		
    		c_S1_i		<= 8'd0;

//			c_gamma		<= 8'sd0;
//			c_beta		<= 8'sd0;

    		c_S2_mean	<= 22'sd0;
    		c_S2_std	<= 8'd0;
//			c_buf_i		<= 9'd0;
    		c_Ex_o		<= 22'sd0;
    		c_Ex2_o		<= 32'd0;
			c_en_S1		<= 1'd0;
//			c_en_S2 	<= 1'd0;
			c_en_wr		<= 1'd0;
			c_en_rd		<= 1'd0;
			c_S2_o		<= 8'd0;
		end

		else
		begin
			c_state		<= n_state;	   
    		c_cnt		<= n_cnt;
    		c_cnt_o		<= n_cnt_o;
    		c_S1_i		<= n_S1_i;

//			c_gamma 	<= n_gamma;
//			c_beta		<= n_beta;

			c_S2_mean	<= n_S2_mean;
			c_S2_std	<= n_S2_std;
//			c_buf_i		<= n_buf_i;
    		c_Ex_o		<= n_Ex_o;	
    		c_Ex2_o		<= n_Ex2_o;
			c_en_S1		<= n_en_S1;
//			c_en_S2 	<= n_en_S2;
			c_en_wr		<= n_en_wr;
			c_en_rd		<= n_en_rd;
			c_S2_o		<= n_S2_o;
		end
	end
	
	

	// n_state 
	always @(*)
	begin
		n_state = c_state;
		case (c_state)
		IDLE :	if (i_start)						n_state = RUN;
		RUN  :  if (S2_done && (c_cnt == COUNT))	n_state = DONE;		
//		RUN  :  if (S2_done && (c_cnt == 6'd39))	n_state = DONE;		// batch = 5
		DONE :  if (S2_done)						n_state = IDLE;
		endcase
	end  


	// n_cnt
	always @(*)
	begin
		n_cnt = c_cnt;
		case (c_state)
		IDLE :										n_cnt = {CNT_WIDTH{1'b0}};
		RUN  : 	if (c_cnt == COUNT)					n_cnt = c_cnt;
				else if (c_en_S1)					n_cnt = c_cnt + 1;
		endcase
	end


	// n_cnt_o
	always @(*)
	begin
		n_cnt_o = c_cnt_o;
		case (c_state)
		IDLE :										n_cnt_o = {CNT_WIDTH{1'b0}};
		RUN  : 	if (c_cnt_o == COUNT)				n_cnt_o = c_cnt_o;
				else if (c_en_rd)					n_cnt_o = c_cnt_o + 1;
		DONE : if (c_cnt_o == COUNT)				n_cnt_o = c_cnt_o;
				else if (c_en_rd)					n_cnt_o = c_cnt_o + 1;

		endcase
	end


	///////////////////////////////////////////////////
	///////////////////////////////////////////////////
	// n_data

	// n_S1_i
	always @(*)
	begin
		n_S1_i = c_S1_i;
		case (c_state)
		IDLE :				n_S1_i = 8'd0;
		RUN  :	if(c_en_S1)	n_S1_i = i_x[((DATA_WIDTH-1) - (8*c_cnt))	-:8];	// [191:184], [183:176], ...
		endcase
	end
	
/*
	// n_gamma
	always @(*)
	begin
		n_gamma = c_gamma;
		case (c_state)
		IDLE :						n_gamma = 8'sd0;
		RUN  :	if (S2_done)		n_gamma = c_gamma;
				else if(Pre_done)	n_gamma = i_gamma[((DATA_WIDTH-1) - (8*c_cnt_o))	-:8];	// [191:184], [183:176], ...
		endcase
	end
*/
	assign gamma_i = (c_en_rd) ? i_gamma[((DATA_WIDTH-1) - (8*c_cnt_o))	-:8] : 8'sd0;

/* 
	// n_beta
	always @(*)
	begin
		n_beta = c_beta;
		case (c_state)
		IDLE :						n_beta = 8'sd0;
		RUN  :	if (S2_done)		n_beta = c_beta;
				else if(Pre_done)	n_beta = i_beta[((DATA_WIDTH-1) - (8*c_cnt_o))	-:8];	// [191:184], [183:176], ...
		endcase
	end
*/

	assign beta_i = (c_en_rd) ? i_beta[((DATA_WIDTH-1) - (8*c_cnt_o))	-:8] : 8'sd0;

//	assign S1_in = (c_en_S1) ? i_x[(191-8*c_cnt)	-:8] : 8'd0;

	// n_Ex_o
	always @(*)
	begin
		n_Ex_o = c_Ex_o;
		if ((c_state == RUN) && S1_done)	n_Ex_o = Ex_out;
	end


	// n_Ex2_o
	always @(*)
	begin
		n_Ex2_o = c_Ex2_o;
		if ((c_state == RUN) && S1_done)	n_Ex2_o = Ex2_out;
	end


/*
	// n_buf_i
	always @(*)
	begin
		n_buf_i = c_buf_i;
		if (c_en_S1)						n_buf_i = buffer_in;
	end
*/


	// n_S2_mean
	always @(*)
	begin
		n_S2_mean = c_S2_mean;
		if (S1_done)						n_S2_mean = Pre_mean;
	end


	// n_S2_std
	always @(*)
	begin
		n_S2_std = c_S2_std;
		if (S1_done)						n_S2_std = Pre_std;
	end


/*
	// n_S2_o
	always @(*)
	begin
		n_S2_o = c_S2_o;
		if (S2_done)						n_S2_o = S2_out;
	end
*/

	//////////////////////////////////////////////////
	//////////////////////////////////////////////////
	// n_enable

	// n_en_S1
	always @(*)
	begin
		n_en_S1 = c_en_S1;
		if (c_cnt % 8 == 7)									n_en_S1 = 1'd0;		// priority : 1. Disable every 8th input
		else if (c_state == IDLE)							n_en_S1 = i_start;	//			  2. 
		else if (S1_done)									n_en_S1 = 1'd1;		//			  3.
	end

/*
	// n_en_S2
	always @(*)
	begin
		n_en_S2 = c_en_S2;
//		if (S1_done)								n_en_S2 = 1'd1;
		if (Pre_done)								n_en_S2 = 1'd1;
		else if (S2_done)							n_en_S2 = 1'd0;
	end
*/

	// n_en_wr
	always @(*)
	begin
		n_en_wr = c_en_wr;
		if (c_cnt % 8 == 7)							n_en_wr = 1'd0;		// priority
		else if (c_en_S1)							n_en_wr = 1'd1;
//		else if ((c_state == RUN) && !buffer_full)	n_en_wr = 1'd1;		
	end


	// n_en_rd -> same as the S2 valid signal
	always @(*)
	begin
		n_en_rd = c_en_rd;
		if (c_cnt % 8 == 7)							n_en_rd = 1'd0;
		else if (S1_done && !buffer_empty)			n_en_rd = 1'd1;
		else if (buffer_empty)						n_en_rd = 1'd0;
	end

/*
	// S1_in 
	assign S1_in = i_x[(191-8*c_cnt)	-:8];	// [191:184], [183:176], ...
                 //i_x[8*c_cnt	+:8]; 			// [7:0], [15:8], ...
*/

	
	//////////////////////////////////////////////////
	//////////////////////////////////////////////////
	// Module Instance
	Stage1 u_Stage1 (
		.i_clk(i_clk),
		.i_rstn(i_rstn),
		.i_valid(c_en_S1),
//		.i_x(S1_in),
		.i_x(c_S1_i),
		.i_alpha(i_alpha),
		.i_inv_n(i_inv_n),
		.o_S1_done(S1_done),
		.o_Ex(Ex_out),
		.o_Ex2(Ex2_out),
		.o_x_norm(buffer_in)
		);


	Input_Buffer #(
		.DEPTH(8),
		.WIDTH(9)
	)
	u_Input_Buffer (
		.i_clk(i_clk),
		.i_rstn(i_rstn),
		.i_wr_en(c_en_wr),
		.i_wr_data(buffer_in),
		.i_rd_en(c_en_rd),
		.o_rd_data(buffer_out),
		.o_full(buffer_full),
		.o_empty(buffer_empty)
	);


	// Preprocess
	Preprocess u_Preprocess (
        .i_valid(S1_done), 
        .i_Ex(Ex_out),
        .i_Ex2(Ex2_out),
        .o_Pre_done(Pre_done),
        .o_mean(Pre_mean),
        .o_std(Pre_std)
    );



	Stage2 #( 
		.COUNT(COUNT),
		.CNT_WIDTH(CNT_WIDTH)
	)
	u_Stage2 (
		.i_clk(i_clk),   
        .i_rstn(i_rstn),  
        .i_valid(c_en_rd),
        .i_x_norm(buffer_out),
        .i_alpha(i_alpha), 
        .i_beta(beta_i),
        .i_gamma(gamma_i),
        .i_mean(c_S2_mean),
        .i_std(c_S2_std),   
        .o_S2_done(S2_done),
        .o_Norm(S2_out)
    );


	/*
	Stage2 u_Stage2 (
		.i_clk(i_clk),   
        .i_rstn(i_rstn),  
        .i_valid(c_en_S2),
        .i_x_norm(buffer_out),
        .i_inv_n(i_inv_n),
        .i_alpha(i_alpha), 
        .i_beta(i_beta),
        .i_gamma(i_gamma),
        .i_Ex(Ex_out),
        .i_Ex2(Ex2_out),   
        .o_S2_done(S2_done),
        .o_Norm(S2_out)
    );
	*/


	//////////////////////////////////////////////////
	//////////////////////////////////////////////////
	// Output
	assign o_done = S2_done;
	assign o_AILayerNorm = (o_done) ? 8'sd0 : S2_out;

	
endmodule
