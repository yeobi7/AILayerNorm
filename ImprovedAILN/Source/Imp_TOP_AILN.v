module Imp_TOP_AILN #(
    parameter N      = 128,
    parameter DATA_W = 8
)(
    input                               i_clk,
    input                               i_rstn,
    input                               i_valid,
    input      signed [DATA_W*N-1:0]    i_x,
    // input       signed   [7:0]           i_alpha,
    // input       signed   [7:0]           i_beta,

    input      signed   [DATA_W*N-1:0]  i_alpha,
    input      signed   [DATA_W*N-1:0]  i_beta,

    output                              o_valid,
    output     signed [DATA_W*N-1:0]    o_norm
);


    localparam EX2_W = DATA_W * 2;

    // =========== Flag ===========
    reg c_s1_busy, n_s1_busy; 
    reg c_s2_busy, n_s2_busy; 


    // =========== F/F ===========
    reg signed [DATA_W*N-1:0]       c_s1_x,       n_s1_x;
    reg signed [DATA_W*N-1:0]       c_s1_alpha,   n_s1_alpha;
    reg signed [DATA_W*N-1:0]       c_s1_beta,    n_s1_beta;
        
            
    reg signed [DATA_W*N-1:0]       c_s2_x,       n_s2_x;
    reg signed [7:0]                c_s2_Ex_i,    n_s2_Ex_i;
    reg        [EX2_W-1:0]          c_s2_Ex2_i,   n_s2_Ex2_i;
    reg signed [DATA_W*N-1:0]       c_s2_alpha,   n_s2_alpha;
    reg signed [DATA_W*N-1:0]       c_s2_beta,    n_s2_beta;
    reg                             c_o_valid,    n_o_valid;
    reg signed [DATA_W*N-1:0]       c_o_norm,     n_o_norm;


    // =========== Control Signal ===========
    wire                        s1_start;
    wire                        s1_done;
    wire                        s2_start;
    wire                        s2_done;
    wire signed [DATA_W-1:0]    Ex_o;
    wire        [EX2_W-1:0]     Ex2_o;
    wire                        Ex_done, Ex2_done;
    wire signed [DATA_W*N-1:0]  affine_o;


    // =========== Sub-Module Instantiations ===========

    // Stage 1
    assign s1_start = c_s1_busy;
    Imp_Ex_Unit  u_Ex  (
        .i_clk(i_clk),
        .i_rstn(i_rstn),
        .i_valid(s1_start), 
        .i_x(c_s1_x), 
        .o_Ex_done(Ex_done), 
        .o_Ex(Ex_o)
    );

    Imp_Ex2_Unit u_Ex2 (
        .i_clk(i_clk),
        .i_rstn(i_rstn),
        .i_valid(s1_start), 
        .i_x(c_s1_x), 
        .o_Ex2_done(Ex2_done), 
        .o_Ex2(Ex2_o)
    );

    assign s1_done = Ex_done & Ex2_done;


    // Stage 2
    assign s2_start = c_s2_busy;
    Imp_Affine_Unit u_Affine (
        .i_clk(i_clk),
        .i_rstn(i_rstn),
        .i_valid(s2_start),
        .i_x(c_s2_x), 
        .i_Ex(c_s2_Ex_i), 
        .i_Ex2(c_s2_Ex2_i),
        .i_alpha(c_s2_alpha), 
        .i_beta(c_s2_beta),
        .o_Affine_done(s2_done), 
        .o_Affine(affine_o)
    );


    // =========== Sequential Logic ===========
 
    always @(posedge i_clk or negedge i_rstn) 
    begin
        if (!i_rstn) 
        begin
            c_s1_busy           <= 0;
            c_s2_busy           <= 0;
            c_s1_x              <= 0;
            c_s1_alpha          <= 0;
            c_s1_beta           <= 0;
            c_s2_x              <= 0;
            c_s2_Ex_i           <= 0;
            c_s2_Ex2_i          <= 0;
            c_s2_alpha          <= 0;
            c_s2_beta           <= 0;
            c_o_valid           <= 0;
            c_o_norm            <= 0;
        end 
        else 
        begin      
            c_s1_busy           <= n_s1_busy;
            c_s2_busy           <= n_s2_busy;
            c_s1_x              <= n_s1_x;
            c_s1_alpha          <= n_s1_alpha;
            c_s1_beta           <= n_s1_beta;
            c_s2_x              <= n_s2_x;
            c_s2_Ex_i           <= n_s2_Ex_i;
            c_s2_Ex2_i          <= n_s2_Ex2_i;
            c_s2_alpha          <= n_s2_alpha;
            c_s2_beta           <= n_s2_beta;
            c_o_valid           <= n_o_valid;
            c_o_norm            <= n_o_norm;
        end
    end


    // =========== Combinational Logic ===========
    // ----------- Control F/F Logic -----------
    always @(*) 
    begin
        // Stage 1 Busy Flag
        n_s1_busy = c_s1_busy;
        if (s1_done)                          n_s1_busy = 1'b0;
        if (i_valid && ~c_s1_busy)            n_s1_busy = 1'b1;

        // Stage 2 Busy Flag
        n_s2_busy = c_s2_busy;
        if (s2_done)                          n_s2_busy = 1'b0;
        if (s1_done)                          n_s2_busy = 1'b1;
    end

    // ----------- Pipeline Register Logic -----------
    always @(*) 
    begin
        n_s1_x       = c_s1_x;
        n_s1_alpha   = c_s1_alpha;
        n_s1_beta    = c_s1_beta;
        n_s2_x       = c_s2_x;
        n_s2_Ex_i    = c_s2_Ex_i;
        n_s2_Ex2_i   = c_s2_Ex2_i;
        n_s2_alpha   = c_s2_alpha;
        n_s2_beta    = c_s2_beta;
        n_o_valid    = 0;
        n_o_norm     = c_o_norm;

        // Stage 1 starts, store the current inputs into the S1 buffer.
        if (i_valid && ~c_s1_busy) 
        begin
            n_s1_x     = i_x;
            n_s1_alpha = i_alpha;
            n_s1_beta  = i_beta;
        end

        // Stage 1 done, transfer the S1 results and the buffered S1 inputs to the Stage 2 register.
        if (s1_done) 
        begin
            n_s2_x     = c_s1_x;
            n_s2_Ex_i  = Ex_o;
            n_s2_Ex2_i = Ex2_o;
            n_s2_alpha = c_s1_alpha;
            n_s2_beta  = c_s1_beta;
        end

        if (s2_done)
        begin
            n_o_valid = 1'b1;
            n_o_norm = affine_o;
        end
    end

    // =========== Output ===========
    assign o_valid = c_o_valid;
    assign o_norm  = c_o_norm;

endmodule