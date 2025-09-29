module tb_Imp_TOP_AILN();

    // --- Parameters ---
    parameter N          = 128;
    parameter DATA_W     = 8;
    parameter CLK_PERIOD = 10;

    // --- Signals ---
    reg                         i_clk;
    reg                         i_rstn;
    reg                         i_valid;
    reg signed [DATA_W*N-1:0]   i_x;
    reg signed [7:0]            i_alpha;
    reg signed [7:0]            i_beta;

    wire                        o_valid;
    wire signed [DATA_W*N-1:0]  o_norm;
    
    integer i;
    integer file_handle;

    // --- DUT Instantiation ---
    Imp_TOP_AILN #(
        .N(N),
        .DATA_W(DATA_W)
    ) u_DUT (
        .i_clk(i_clk),
        .i_rstn(i_rstn),
        .i_valid(i_valid),
        .i_x(i_x),
        .i_alpha(i_alpha),
        .i_beta(i_beta),
        .o_valid(o_valid),
        .o_norm(o_norm)
    );

    // --- Clock Generation ---
    always #(CLK_PERIOD/2) i_clk = ~i_clk;

    // --- Stimulus ---
    initial begin
        // 1. Reset
        i_clk   = 1'b0; i_rstn  = 1'b0; i_valid = 1'b0; i_x = 'd0; i_alpha = 'd0; i_beta  = 'd0;
        #(CLK_PERIOD * 2); i_rstn = 1'b1;

        // 2. 파일 열기
        file_handle = $fopen("top_norm_output.txt", "w");
        
        // --- 배치 A 시작 및 결과 저장 ---
        // Apply Batch A
        for (i = 0; i < N; i = i + 1) begin i_x[i*DATA_W +: DATA_W] = i; end
        i_alpha = 2; i_beta  = 2;
        @(posedge i_clk); i_valid = 1'b1; @(posedge i_clk); i_valid = 1'b0;
        
        // Wait for the first o_valid
        @(posedge o_valid);
        @(posedge i_clk); // o_norm 레지스터가 업데이트될 때까지 한 사이클 대기
        
        // Write Batch A result to file
        $fdisplay(file_handle, "--- Result for Batch A (inputs 0 to 127) ---");
        for (i = 0; i < N; i = i + 1) begin
            $fdisplay(file_handle, "o_norm[%3d] = %d", i, $signed(o_norm[i*DATA_W +: DATA_W]));
        end

        // --- 배치 B 시작 및 결과 저장 ---
        // Apply Batch B (Wait 14 cycles, total 16 from last valid)
        #(CLK_PERIOD * 14);
        for (i = 0; i < N; i = i + 1) begin i_x[i*DATA_W +: DATA_W] = i - 127; end
        i_alpha = 1; i_beta  = -1;
        @(posedge i_clk); i_valid = 1'b1; @(posedge i_clk); i_valid = 1'b0;
        
        // Wait for the second o_valid
        @(posedge o_valid);
        @(posedge i_clk); // o_norm 레지스터가 업데이트될 때까지 한 사이클 대기

        // Write Batch B result to file
        $fdisplay(file_handle, "\n--- Result for Batch B (inputs -127 to 0) ---");
        for (i = 0; i < N; i = i + 1) begin
            $fdisplay(file_handle, "o_norm[%3d] = %d", i, $signed(o_norm[i*DATA_W +: DATA_W]));
        end

        // --- 시뮬레이션 종료 ---
        $fclose(file_handle);
        #(CLK_PERIOD * 5);
    end
    

endmodule