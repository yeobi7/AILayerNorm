module tb_Imp_Ex2_Unit ();

	// Parameters 
    parameter N      = 128;
    parameter DATA_W = 8;
    parameter EX2_W  = 16;

	reg 						i_clk;
	reg							i_rstn;
	reg							i_valid;
	reg	 signed	[DATA_W*N-1:0]	i_x;

	wire						o_Ex2_done;
	wire signed	[EX2_W-1:0]		o_Ex2;

	
	always #5 i_clk = ~i_clk;

	
    Imp_Ex2_Unit #(
        .N(N),
        .DATA_W(DATA_W),
        .EX2_W(EX2_W)
	) u_Imp_Ex2_Unit (
        .i_clk(i_clk),
        .i_rstn(i_rstn),
        .i_valid(i_valid),
        .i_x(i_x),
        .o_Ex2_done(o_Ex2_done),
        .o_Ex2(o_Ex2)
    );
			
	integer i;
	
	initial
	begin

		i_clk = 1'd0;	i_rstn = 1'd0;	i_valid = 1'd0;		
		#1 i_rstn = 1'd1;	#1 i_rstn = 1'd0; 	#2 i_rstn = 1'd1;	
		

		for (i = 0; i < N; i = i + 1) begin
            i_x[i*DATA_W +: DATA_W] = i;
        end

		
		@(posedge i_clk);
        i_valid = 1'b1;
        
        @(posedge i_clk);
        i_valid = 1'b0;
		


	end

endmodule

