module tb_Ex_Unit ();
	reg 				i_clk;
	reg					i_rstn;
	reg					i_valid;
	reg	 signed	[8:0]	i_x;
	reg  		[1:0]	i_alpha;
	reg	 		[7:0]	i_inv_n;

	wire				o_Ex_done;
	wire signed	[7:0]	o_Ex;

	
	always #5 i_clk = ~i_clk;


	Ex_Unit u_Ex_Unit (
		.i_clk(i_clk),
        .i_rstn(i_rstn),
        .i_valid(i_valid), 
        .i_x(i_x),
        .i_alpha(i_alpha),
        .i_inv_n(i_inv_n),
        .o_Ex_done(o_Ex_done),
        .o_Ex(o_Ex)
    );
			

	initial
	begin
		i_clk = 1'd0;	i_rstn = 1'd0;	i_valid = 1'd0;		
		#1 i_rstn = 1'd1;	#1 i_rstn = 1'd0; 	#2 i_rstn = 1'd1;
		
		#5 i_valid = 1'd1;	i_alpha = 2'd2;		i_inv_n = 8'd32;
		
		
		#10 i_x = 8'sd1;
		#10 i_x = 8'sd2; 	
		#10 i_x = 8'sd3;	
		#10 i_x = 8'sd4;	
		#10 i_x = 8'sd5;			
		#10 i_x = 8'sd6;	
		#10 i_x = 8'sd7;			
		#10 i_x = 8'sd8;	



	end

endmodule
