module tb_Imp_Ex_Unit ();
	reg 				i_clk;
	reg					i_rstn;
	reg					i_valid;
	reg	 signed	[7:0]	i_x;

	wire				o_Ex_done;
	wire signed	[8:0]	o_Ex;

	
	always #5 i_clk = ~i_clk;

	Imp_Ex_Unit u_Imp_Ex_Unit (
		.i_clk(i_clk),
        .i_rstn(i_rstn),
        .i_valid(i_valid), 
        .i_x(i_x),
        .o_Ex_done(o_Ex_done),
        .o_Ex(o_Ex)
    );
			

	initial
	begin
		i_clk = 1'd0;	i_rstn = 1'd0;	i_valid = 1'd0;		
		#1 i_rstn = 1'd1;	#1 i_rstn = 1'd0; 	#2 i_rstn = 1'd1;
		
		#5 i_valid = 1'd1;	
		
		
		#10 i_x = -8'sd1;
		#10 i_x = -8'sd2; 	
		#10 i_x = -8'sd3;	
		#10 i_x = -8'sd4;	
		#10 i_x = -8'sd5;			
		#10 i_x = -8'sd6;	
		#10 i_x = -8'sd7;			
		#10 i_x = -8'sd7;	



	end

endmodule

