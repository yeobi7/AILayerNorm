//==========================================================================
// Module: Imp_Ex2_Unit
// Description: Squared and accumulated (Ex2) N 8-bit inputs, and averaged.
// Using Divider ip 
//==========================================================================

/*
module Imp_Ex2_Unit #(
    parameter N = 8                  // # of input
)(
    input                   i_clk,
    input                   i_rstn,
    input                   i_valid,
    input  signed [7:0]     i_x,         // sign bit + int8

    output                  o_Ex2_done,
    output signed [2*N-1:0] o_Ex2        // EX2_WIDTH
);

    // FSM state
    localparam IDLE   = 3'd0;
    localparam ACC    = 3'd1;
    localparam DIVIDE = 3'd2;
    localparam DONE   = 3'd3;

    // bit width
    localparam DATA_W           = 8;
    localparam SQUARED_WIDTH    = (2*DATA_W - 1);
    localparam CNT_WIDTH        = $clog2(N);
    localparam ACC_WIDTH        = SQUARED_WIDTH + CNT_WIDTH;
    localparam EX2_WIDTH        = SQUARED_WIDTH;
    localparam DOUT_WIDTH       = ACC_WIDTH + CNT_WIDTH;

    /////////////// F/F /////////////////
    reg         [2:0]               n_state,   c_state;
    reg         [ACC_WIDTH-1:0]     n_acc,     c_acc;
    reg         [CNT_WIDTH-1:0]     n_cnt_acc, c_cnt_acc;
    reg signed  [EX2_WIDTH-1:0]     n_Ex2,     c_Ex2;

    // ------------------------------ Hybrid Square ------------------------------
    wire [SQUARED_WIDTH-1:0]   squared_x;
    wire [3:0]                 H, L;
    wire [7:0]                 H_sq, L_sq, HxL;

    // Separate the absolute value of an 8-bit input into upper (H) and lower (L) 4 bits
    assign H = i_x[7:4];
    assign L = i_x[3:0];

    // 4bit square lut instance
    Imp_mult4b_lut u_H_sq (.addr(H), .data(H_sq));
    Imp_mult4b_lut u_L_sq (.addr(L), .data(L_sq));

    // 4x4 mult
    assign HxL = H * L;

    // X^2 = 256*H^2 + 32*H*L + L^2
    assign squared_x = (H_sq << 8) + (HxL << 5) + L_sq;
    // --------------------------------------------------------------------------


    /////////////// Divider IP port ////////////////
    wire                            div_Acc_tvalid;
    wire                            div_Acc_tready;
    wire signed [ACC_WIDTH-1:0]     div_Acc_tdata;

    wire                            div_N_tvalid;
    wire                            div_N_tready;
    wire        [CNT_WIDTH-1:0]     div_N_tdata;

    wire                            div_Ex2_tvalid;
    wire                            div_Ex2_tready;
    wire        [DOUT_WIDTH-1:0]    div_Ex2_tdata;
    wire signed [EX2_WIDTH-1:0]     div_quotient;


    // =================== Xilinx Divider Generator IP ===================
    div_gen_0 U_DIV (
        .aclk(i_clk),

        // Dividend Channel (Acc)
        .s_axis_dividend_tvalid(div_Acc_tvalid),
        .s_axis_dividend_tready(div_Acc_tready),
        .s_axis_dividend_tdata (div_Acc_tdata),

        // Divisor Channel (N)
        .s_axis_divisor_tvalid (div_N_tvalid),
        .s_axis_divisor_tready (div_N_tready),
        .s_axis_divisor_tdata  (div_N_tdata),

        // Output Channel (Ex)
        .m_axis_dout_tvalid    (div_Ex2_tvalid),
        .m_axis_dout_tready    (div_Ex2_tready),
        .m_axis_dout_tdata     (div_Ex2_tdata)
    );
    //=====================================================================

    // Extract only the quotient from the IP output
    assign div_quotient = div_Ex2_tdata[EX2_WIDTH-1:0];


    // F/F (상태 및 데이터 레지스터)
    always @(posedge i_clk or negedge i_rstn)
    begin
        if (!i_rstn)
        begin
            c_state   <= IDLE;
            c_acc     <= 0;
            c_cnt_acc <= 0;
            c_Ex2     <= 0;
        end
        else
        begin
            c_state   <= n_state;
            c_acc     <= n_acc;
            c_cnt_acc <= n_cnt_acc;
            c_Ex2     <= n_Ex2;
        end
    end


    // n_state
    always @(*)
    begin
        n_state = c_state;
        case (c_state)
            IDLE   : if (i_valid)                                           n_state = ACC;
            ACC    : if (c_cnt_acc == N-1)                                  n_state = DIVIDE;
            DIVIDE : if (div_Acc_tready && div_N_tready && div_Ex2_tvalid)  n_state = DONE;
            DONE   :                                                        n_state = IDLE;
        endcase
    end
    
    // n_cnt_acc
    always @(*)
    begin
        n_cnt_acc = c_cnt_acc;
        case (c_state)
            IDLE : n_cnt_acc = 0;
            ACC  : if (i_valid) n_cnt_acc = c_cnt_acc + 1;
        endcase
    end

    // n_acc
    always @(*)
    begin
        n_acc = c_acc;
        case (c_state)
            IDLE : n_acc = 0;
            ACC  : if (i_valid) n_acc = c_acc + squared_x;
        endcase
    end

    // n_Ex2
    always @(*)
    begin
        n_Ex2 = c_Ex2;
        case (c_state)
            IDLE : n_Ex2 = 0;
            DONE : n_Ex2 = div_quotient;
        endcase
    end


    // Divider IP control signal
    assign div_Acc_tvalid  = (c_state == DIVIDE);
    assign div_N_tvalid    = (c_state == DIVIDE);
    assign div_Acc_tdata   = c_acc;
    assign div_N_tdata     = N;
    assign div_dout_tready = 1'b1;

    // 최종 출력
    assign o_Ex2      = c_Ex2;
    assign o_Ex2_done = (c_state == DONE);

endmodule
*/


//==========================================================================
// Module: Imp_Ex2_Unit
// Description: Squared and accumulated (Ex2) N 8-bit inputs, and averaged.
// Divide Using shift  
//==========================================================================

module Imp_Ex2_Unit #(
    parameter N = 8                  // # of input
)(
    input                   i_clk,
    input                   i_rstn,
    input                   i_valid,
    input  signed [7:0]     i_x,         // sign bit + int8

    output                  o_Ex2_done,
    output        [15:0] o_Ex2        // EX2_WIDTH
);

    // FSM state
    localparam IDLE   = 2'd0;
    localparam ACC    = 2'd1;
    localparam DIVIDE = 2'd2;
    localparam DONE   = 2'd3;
 

    // bit width
    localparam DATA_W           = 8;
    localparam SQUARED_WIDTH    = (2*DATA_W - 1);
    localparam CNT_WIDTH        = $clog2(N);
    localparam SHIFT            = CNT_WIDTH;
    localparam ROUND_BIAS       = (1 << (SHIFT-1));
    localparam ACC_WIDTH        = SQUARED_WIDTH + CNT_WIDTH;
    localparam EX2_WIDTH        = SQUARED_WIDTH;
    localparam DOUT_WIDTH       = ACC_WIDTH + CNT_WIDTH;


    /////////////// F/F /////////////////
    reg         [2:0]               n_state,   c_state;
    reg         [ACC_WIDTH-1:0]     n_acc,     c_acc;
    reg         [CNT_WIDTH-1:0]     n_cnt_acc, c_cnt_acc;
    reg signed  [EX2_WIDTH-1:0]     n_Ex2,     c_Ex2;


    // ----------------------------------- Hybrid Square -----------------------------------
    wire [SQUARED_WIDTH-1:0]    squared_x;
    wire [3:0]                  H, L;
    wire [7:0]                  H_sq, L_sq, HxL;
    wire [7:0]                  x_abs;


    // Separate the absolute value of an 8-bit input into upper (H) and lower (L) 4 bits
    assign x_abs = i_x[7] ? (~i_x + 8'd1) : i_x;        // = (i_x < 0) ? -i_x : i_x
    assign H = x_abs[7:4];
    assign L = x_abs[3:0];


    // 4bit square lut instance
    Imp_mult4b_lut u_H_sq (.addr(H), .data(H_sq));
    Imp_mult4b_lut u_L_sq (.addr(L), .data(L_sq));


    // 4x4 mult
    assign HxL = H * L;


    // X^2 = 256*H^2 + 32*H*L + L^2
    assign squared_x = (H_sq << 8) + (HxL << 5) + L_sq;
    // ------------------------------------------------------------------------------------


    // Rounding Logic
    wire [ACC_WIDTH:0]      acc_rounded;
    wire [EX2_WIDTH-1:0]    avg_ex2;        

    assign acc_rounded = {1'b0, c_acc} + ROUND_BIAS;        // 1비트 여유
    assign avg_ex2     = acc_rounded >> SHIFT;              // = floor((x+b)/2^k)


    // F/F
    always @(posedge i_clk or negedge i_rstn)
    begin
        if (!i_rstn)
        begin
            c_state   <= IDLE;
            c_acc     <= 0;
            c_cnt_acc <= 0;
            c_Ex2     <= 0;
        end
        else
        begin
            c_state   <= n_state;
            c_acc     <= n_acc;
            c_cnt_acc <= n_cnt_acc;
            c_Ex2     <= n_Ex2;
        end
    end


    // n_state
    always @(*)
    begin
        n_state = c_state;
        case (c_state)
            IDLE   : if (i_valid)                n_state = ACC;
            ACC    : if (c_cnt_acc == N-1)       n_state = DIVIDE;
            DIVIDE :                             n_state = DONE;          
            DONE   :                             n_state = IDLE;
        endcase
    end
    
    // n_cnt_acc
    always @(*)
    begin
        n_cnt_acc = c_cnt_acc;
        case (c_state)
            IDLE :                  n_cnt_acc = 0;
            ACC  : if (i_valid)     n_cnt_acc = c_cnt_acc + 1;
        endcase
    end

    // n_acc
    always @(*)
    begin
        n_acc = c_acc;
        case (c_state)
            IDLE :                  n_acc = 0;
            ACC  : if (i_valid)     n_acc = c_acc + squared_x;
        endcase
    end

    // n_Ex2
    always @(*)
    begin
        n_Ex2 = c_Ex2;
        case (c_state)
            IDLE   :                  n_Ex2 = 0;
            DIVIDE :                  n_Ex2 = avg_ex2;
        endcase
    end




    // 최종 출력
    assign o_Ex2      = (c_state == DONE) ? c_Ex2 : 0;
    assign o_Ex2_done = (c_state == DONE);

endmodule