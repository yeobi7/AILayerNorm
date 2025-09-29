
module tb_Imp_Affine_Unit();

    parameter N      = 128;
    parameter DATA_W = 8; 
    
    // DUT signals
    reg                          i_clk;
    reg                          i_rstn;
    reg                          i_valid;
    reg signed  [DATA_W*N-1:0]   i_x;
    reg signed  [7:0]            i_Ex;
    reg         [15:0]           i_Ex2;
    reg signed  [7:0]            i_alpha;
    reg signed  [7:0]            i_beta;
    
    wire                         o_Affine_done;
    wire signed [DATA_W*N-1:0]   o_Affine;
    
    always #5 i_clk = ~i_clk;

    Imp_Affine_Unit #(
            .N(N),
            .DATA_W(DATA_W)
    ) u_Imp_Affine_Unit (
            .i_clk(i_clk),
            .i_rstn(i_rstn),
            .i_valid(i_valid),
            .i_x(i_x),
            .i_Ex(i_Ex),
            .i_Ex2(i_Ex2),
            .i_alpha(i_alpha),
            .i_beta(i_beta),
            .o_Affine_done(o_Affine_done),
            .o_Affine(o_Affine)
    );


    integer i;
    integer file_handle;

    initial
    begin
        i_clk = 1'd0;   i_rstn = 1'd0;  i_valid = 1'd0;
        #1 i_rstn = 1'd1;   #1 i_rstn = 1'd0;   #2 i_rstn = 1'd1;   

        #5 i_Ex = 8'sd63;    i_Ex2 = 16'd5398;     i_alpha = 8'sd2;    i_beta = 8'sd2;


    	for (i = 0; i < N; i = i + 1) begin
            i_x[i*DATA_W +: DATA_W] = i;
        end

		
		@(posedge i_clk);
        i_valid = 1'b1;
        
        @(posedge i_clk);
        i_valid = 1'b0;
    
        wait (o_Affine_done == 1'b1);

        file_handle = $fopen("affine_output.txt", "w");
        if (file_handle) begin
            for (i = 0; i < N; i = i + 1) begin
                $fdisplay(file_handle, "o_Affine[%3d] = %d", i, $signed(o_Affine[i*DATA_W +: DATA_W]));
            end
            $fclose(file_handle);
            $display("Results written to affine_output.txt");
        end else begin
            $display("ERROR: Could not open file for writing.");
        end

        #1000;
        $finish;

    end



endmodule