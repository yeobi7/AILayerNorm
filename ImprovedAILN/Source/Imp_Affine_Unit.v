//==========================================================================
// Module: Imp_Affine_Unit
// Description: Calculates the var and std, Normalization with alpha, beta 
//==========================================================================

module Imp_Affine_Unit #(
    parameter N      = 128,
    parameter DATA_W = 8
)(
    input                                i_clk,
    input                                i_rstn,
    input                                i_valid,
    input       signed   [DATA_W*N-1:0]  i_x,
    input       signed   [7:0]           i_Ex,
    input                [15:0]          i_Ex2,
    // input       signed   [7:0]           i_alpha,
    // input       signed   [7:0]           i_beta,

    input       signed   [DATA_W*N-1:0]  i_alpha,
    input       signed   [DATA_W*N-1:0]  i_beta,

    output                               o_Affine_done,
    output reg  signed   [DATA_W*N-1:0]  o_Affine         
);


    // FSM state
    localparam  IDLE = 3'd0; 
    localparam  PREP = 3'd1;
    localparam  SQRT = 3'd2;
    localparam  CALC = 3'd3;    // calculate affine scale
    localparam  NORM = 3'd4;
    localparam  DONE = 3'd5;


    // bit width
    localparam  CHUNK_SIZE      = 16;
    localparam  NUM_CHUNK       = N / CHUNK_SIZE;
    localparam  SQUARED_WIDTH   = (2*DATA_W - 1);
    localparam  CNT_WIDTH       = $clog2(N);
    localparam  INV_STD_WIDTH   = 14; 
    localparam  SHIFT           = 14;   // INV_STD_WIDTH - ALPHA_WIDTH - OUTPUT_WIDTH = 14 + 8 - 8 = 14

    


    //////////////////// F/F //////////////////// 
    reg         [2:0]               n_state,        c_state;
    reg         [CNT_WIDTH-1:0]     n_cnt,          c_cnt;
//    reg         [15:0]              n_var,          c_var;
    reg         [7:0]               n_std,          c_std;
//    reg         [INV_STD_WIDTH:0]   n_inv_std,      c_inv_std;
    reg signed  [7:0]               n_affine_scale, c_affine_scale;
//    reg signed  [15:0]              n_norm,         c_norm;



    //////////////////// Datapath Wires ////////////////////
    wire signed [DATA_W-1:0]            selected_x[CHUNK_SIZE-1:0];
    wire signed [DATA_W-1:0]            selected_alpha[CHUNK_SIZE-1:0];
    wire signed [DATA_W-1:0]            selected_beta[CHUNK_SIZE-1:0];
    wire signed [DATA_W-1:0]            normalized_chunk[CHUNK_SIZE-1:0];
    wire        [DATA_W*CHUNK_SIZE-1:0] flattened_chunk;



    // ------------------ Sequential Logic ------------------
    always @(posedge i_clk or negedge i_rstn)
    begin
        if(!i_rstn)
        begin
            c_state         <= IDLE;
            c_cnt           <= 0;
    //        c_var           <= 0;
            c_std           <= 0;
    //        c_inv_std       <= 0;
            c_affine_scale  <= 0;
    //        c_norm          <= 0;
        end
        else
        begin
            c_state         <= n_state;
            c_cnt           <= n_cnt;
    //        c_var           <= n_var;
            c_std           <= n_std;
    //        c_inv_std       <= n_inv_std;
            c_affine_scale  <= n_affine_scale;
    //        c_norm          <= n_norm; 
        end                     
    end                         
                                
    // ------------------------------------------------------


    // ----------------------------------- Hybrid Square -----------------------------------
    wire [SQUARED_WIDTH-1:0]    squared_Ex;
    wire [3:0]                  H, L;
    wire [7:0]                  H_sq, L_sq, HxL;
    wire [7:0]                  Ex_abs;


    // Separate the absolute value of an 8-bit input into upper (H) and lower (L) 4 bits
    assign Ex_abs = i_Ex[7] ? (~i_Ex + 8'd1) : i_Ex;        // = (i_Ex < 0) ? -i_Ex : i_Ex
    assign H = Ex_abs[7:4];
    assign L = Ex_abs[3:0];


    // 4bit square lut instance
    Imp_mult4b_lut u_H_sq (.addr(H), .data(H_sq));
    Imp_mult4b_lut u_L_sq (.addr(L), .data(L_sq));


    // 4x4 mult
    assign HxL = H * L;


    // X^2 = 256*H^2 + 32*H*L + L^2
    assign squared_Ex = (H_sq << 8) + (HxL << 5) + L_sq;
    // ------------------------------------------------------------------------------------


    // ---------------------- n_state ----------------------
    always @(*)
    begin
        n_state = c_state;
        case(c_state)
            IDLE    : if (i_valid)                  n_state = PREP;
            PREP    :                               n_state = SQRT;
            SQRT    : if (std_done)                 n_state = CALC;
            CALC    :                               n_state = NORM;
            NORM    : if (c_cnt == NUM_CHUNK-1)     n_state = DONE; // if or not ?
            DONE    :                               n_state = IDLE;
        endcase
    end



    // ---------------------- var ----------------------
    // always @(*)
    // begin
    //     n_var = c_var;
    //     case(c_state)
    //         PREP    :                   n_var = i_Ex2 - squared_Ex;
    //     endcase
    // end

    // var - non f/f
    wire [15:0] var;
    //assign var = (c_state == PREP) ? i_Ex2 - squared_Ex : 0;
    assign var = i_Ex2 - squared_Ex;

  


    // ---------------------- std ----------------------

    // Imp_cal_sqrt instance -> Calc std => sqrt(var)
    
    wire std_start;
    assign std_start = (c_state == PREP);

    wire [7:0]  std_o;
    Imp_cal_sqrt u_std (
        .i_clk(i_clk),
        .i_rstn(i_rstn),
        .i_valid(std_start),
        .i_data(var),
        .o_sqrt(std_o),
        .o_done(std_done)
    );



    always @(*)
    begin
        n_std = c_std;
        case(c_state)
            SQRT    :   if (std_done)          n_std = std_o;
        endcase
    end



    // inv_std.mem instance -> Calc 1/std using .mem file (lut)
    wire [INV_STD_WIDTH-1:0] inv_std_o;
    Imp_inv_std_lut u_imp_inv_std_lut(
        .addr(c_std),
        .data(inv_std_o)
    );



    // ---------------------- n_cnt ----------------------
    always @(*) begin
        n_cnt = c_cnt;
        case(c_state)
            IDLE    :                           n_cnt = 0;
            NORM    : if (c_cnt < NUM_CHUNK-1)  n_cnt = c_cnt + 1;
        endcase
    end


    // ---------------------- n_inv_std ----------------------
    // always @(*)
    // begin
    //     n_inv_std = c_inv_std;
    //     case(c_state)
    //         NORM    :                   n_inv_std = inv_std_o;
    //     endcase
    // end


    // alpha * 1/std 
    wire signed [21:0]   temp_mul;           // alpha * 1/std -> Q22
    wire signed [22:0]   temp_round;         
    wire signed [7:0]    affine_scale_q;    

    // //assign temp_mul = i_alpha * $signed({1'b0, c_inv_std}); // Q22
    // assign temp_mul = i_alpha * $signed({1'b0, inv_std_o}); // Q22
    // assign temp_round = temp_mul + (1<<(SHIFT-1));     // 2^(SHIFT-1) = 2^13 = 8192
    // assign affine_scale_q = temp_round[21:14];  

    // -------------------------------------------------------



    // ---------------------- n_affine_scale ----------------------
    always @(*)
    begin
        n_affine_scale= c_affine_scale;
        case(c_state)
            CALC    :                   n_affine_scale = affine_scale_q;
        endcase
    end
    // -------------------------------------------------------------



//     // ---------------------- Normalization ----------------------
//     wire signed [8:0]  x_minus_ex = i_x - i_Ex;
//     wire signed [16:0] temp_normalized_scale = x_minus_ex * c_affine_scale;


//     wire signed [15:0] temp_final = temp_normalized_scale[15:0] + {{8{i_beta[7]}}, i_beta}; 

// /*    
//     // n_normalized
//     always @(*)
//     begin
//         n_norm = c_norm;
//         case (c_state)
//             NORM    :                   n_norm = temp_final;
//         endcase
//     end
// */

//     // -----------------------------------------------------------



/*
    // ---------------------- n_var ----------------------
    always @(*)
    begin
        n_cnt = c_cnt;
        case(c_state)
            NORM    :                   n_cnt = c_cnt + 1;
        endcase
    end
*/



    /////////////// Select a 16-element chunk from the input vector ///////////////
    genvar i_selected;
    generate
        for(i_selected=0; i_selected < CHUNK_SIZE; i_selected=i_selected+1) 
        begin : gen_select
            assign selected_x    [i_selected] = i_x    [(c_cnt * CHUNK_SIZE * DATA_W) + (i_selected * DATA_W) +: DATA_W];
            assign selected_alpha[i_selected] = i_alpha[(c_cnt * CHUNK_SIZE * DATA_W) + (i_selected * DATA_W) +: DATA_W];
            assign selected_beta [i_selected] = i_beta [(c_cnt * CHUNK_SIZE * DATA_W) + (i_selected * DATA_W) +: DATA_W];
        end
    endgenerate



    /////////////// 16-wide Parallel Normalization Units ///////////////
    genvar i_norm;
    generate
        for (i_norm = 0; i_norm < CHUNK_SIZE; i_norm = i_norm + 1) 
        begin : gen_norm

            // x - E[x]
            wire signed [8:0]  x_minus_ex = selected_x[i_norm] - i_Ex;
            // * alpha
            wire signed [16:0] temp_mul_alpha = x_minus_ex * selected_alpha[i_norm];
            // * (1/std)
            wire signed [31:0] temp_mul_inv_std = temp_mul_alpha * $signed({1'b0, inv_std_o});
            // Rounding
            wire signed [32:0] temp_rounded = temp_mul_inv_std + (1 << (SHIFT - 1));
            // Divide (Shift)
            wire signed [18:0] scaled_result = temp_rounded >> SHIFT;
            // Add beta
            assign normalized_chunk[i_norm] = scaled_result + selected_beta[i_norm];
        end
    endgenerate

    /////////////// Flattening: 2D chunk -> 1D vector ///////////////
    genvar k;
    generate
        for (k=0; k < CHUNK_SIZE; k=k+1)
        begin : gen_flatten
            assign flattened_chunk[(k+1)*DATA_W-1 -: DATA_W] = normalized_chunk[k];
        end
    endgenerate




    // output
    always @(posedge i_clk) begin
        if (c_state == NORM) 
        begin
            o_Affine[(c_cnt * CHUNK_SIZE * DATA_W) +: CHUNK_SIZE * DATA_W] <= flattened_chunk;
        end
    end


    assign o_Affine_done = (c_state == DONE) ? 1 : 0;



endmodule


