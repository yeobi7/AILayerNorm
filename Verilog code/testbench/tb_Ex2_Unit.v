module tb_Ex2_Unit ();
	reg 		i_clk;
	reg			i_rstn;
	reg			i_valid;
	reg	 [8:0]	i_x;
	reg  [1:0]	i_alpha;
	reg	 [7:0]	i_inv_n;

	wire		o_Ex2_done;
	wire [7:0]	o_Ex2;

	
	always #5 i_clk = ~i_clk;


	Ex2_Unit u_Ex2_Unit (
		.i_clk(i_clk),
        .i_rstn(i_rstn),
        .i_valid(i_valid), 
        .i_x(i_x),
        .i_alpha(i_alpha),
        .i_inv_n(i_inv_n),
        .o_Ex2_done(o_Ex2_done),
        .o_Ex2(o_Ex2)
    );
			

	initial
	begin
		i_clk = 1'd0;	i_rstn = 1'd0;	i_valid = 1'd0;		
		#1 i_rstn = 1'd1;	#1 i_rstn = 1'd0; 	#2 i_rstn = 1'd1;
		
		#5 i_alpha = 2'd2;	i_inv_n = 8'd32;
		
		
		#10 i_valid = 1'd1; i_x = 8'd1;
		#10 i_x = 8'd2; 	
		#10 i_x = 8'd3;	
		#10 i_x = 8'd4;	
		#10 i_x = 8'd5;			
		#10 i_x = 8'd6;	
		#10 i_x = 8'd7;			
		#10 i_x = 8'd8;	
		#10 i_x = 8'd9;
		#10 i_x = 8'd10;
		#10 i_x = 8'd11;
		#10 i_x = 8'd12;
		#10 i_x = 8'd13;
		#10 i_x = 8'd14;
        #10 i_x = 8'd15;		
        #10 i_x = 8'd16;


	end

endmodule
