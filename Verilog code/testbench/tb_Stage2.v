module tb_Stage2 ();
	reg						i_clk;
	reg						i_rstn;
	reg						i_valid;
	reg				[ 8:0]	i_x_norm;
	reg				[ 7:0]	i_gamma;
    reg				[ 7:0]	i_beta;
    reg		signed	[21:0]  i_Ex;
    reg   			[31:0]  i_Ex2;            
    reg   			[ 1:0]  i_alpha;   	
    reg   			[ 7:0]  i_inv_n;    


    wire         		   	o_S2_done;         
    wire	signed	[ 7:0]  o_Norm;


	always #5 i_clk = ~i_clk;


	Stage2 u_Stage2(
		.i_clk(i_clk),
		.i_rstn(i_rstn),
		.i_valid(i_valid),
		.i_x_norm(i_x_norm),
		.i_gamma(i_gamma),
		.i_beta(i_beta),
		.i_Ex(i_Ex),
		.i_Ex2(i_Ex2),
		.i_alpha(i_alpha),
		.i_inv_n(i_inv_n),
		.o_S2_done(o_S2_done),
		.o_Norm(o_Norm)
	);


	initial 
	begin
		i_x_norm = 9'sd0;		i_alpha = 2'd0;		i_gamma = 8'd0;		i_beta = 8'd0;		i_Ex = 8'sd0;
        
        i_Ex2 = 8'd0;		i_inv_n = 8'd0;		i_clk = 1'd0;		i_rstn = 1'd0;		i_valid =  1'd0;	

		#1 i_rstn = 1'd1;	#1 i_rstn = 1'd0;	#2 i_rstn = 1'd1;

		#5 	i_valid = 1'd1;		i_alpha = 2'd2;		i_inv_n = 8'd32;	i_gamma = 8'd100;		i_beta = 8'd10;
			i_Ex = 22'sd60;		i_Ex2 = 32'd4654;

		
		#10 i_x_norm = 9'sd44;
		#10 i_x_norm =-9'sd81;
		#10 i_x_norm =-9'sd11; 
		#10 i_x_norm = 9'sd64; 
		#10 i_x_norm =-9'sd61; 
		#10 i_x_norm = 9'sd123; 
		#10 i_x_norm = 9'sd67; 
		#10 i_x_norm =-9'sd25; 
	end
endmodule

