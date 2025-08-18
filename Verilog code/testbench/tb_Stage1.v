module tb_Stage1 ();
	reg					i_clk;
	reg					i_rstn;
	reg					i_valid;
	reg			[7:0]	i_x;
	reg			[1:0]	i_alpha;
	reg			[7:0]	i_inv_n;

	wire				o_S1_done;
	wire		[21:0]	o_Ex;
	wire		[31:0]	o_Ex2;
	wire signed	[ 8:0]	o_x_norm;


	always #5 i_clk = ~i_clk;


	Stage1 u_Stage1(
		.i_clk(i_clk),
		.i_rstn(i_rstn),
		.i_valid(i_valid),
		.i_x(i_x),
		.i_alpha(i_alpha),
		.i_inv_n(i_inv_n),
		.o_S1_done(o_S1_done),
		.o_Ex(o_Ex),
		.o_Ex2(o_Ex2),
		.o_x_norm(o_x_norm)
	);


	initial 
	begin
		i_clk = 1'd0;	i_rstn = 1'd0;	i_valid =  1'd0;
		#1 i_rstn = 1'd1;	#1 i_rstn = 1'd0;	#2 i_rstn = 1'd1;

		#5 i_valid = 1'd1;	i_alpha = 2'd2;		i_inv_n = 8'd32;

		#10 i_x = 8'd172;
		#10 i_x = 8'd47;
		#10 i_x = 8'd117; 
		#10 i_x = 8'd192; 
		#10 i_x = 8'd67; 
		#10 i_x = 8'd251; 
		#10 i_x = 8'd195; 
		#10 i_x = 8'd103; 
	end
endmodule
