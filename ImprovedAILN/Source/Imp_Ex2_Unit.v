// //==========================================================================
// // Module: Imp_Ex2_Unit
// // Description: Squared and accumulated (Ex2) N 8-bit inputs, and averaged.
// // Divide Using shift  
// //==========================================================================

// module Imp_Ex2_Unit #(
//     parameter N         = 128,                  // # of input
//     parameter DATA_W    = 8

// )(
//     input                       i_clk,
//     input                       i_rstn,
//     input                       i_valid,
//     input  signed [DATA_W*N:0]  i_x,         // sign bit + int8

//     output                      o_Ex2_done,
//     output        [15:0]        o_Ex2        // EX2_WIDTH
// );

//     // FSM state
//     localparam IDLE   = 2'd0;
//     localparam ACC    = 2'd1;
//     localparam DIVIDE = 2'd2;
//     localparam DONE   = 2'd3;
 

//     // bit width
//     localparam SQUARED_WIDTH    = (2*DATA_W - 1);
//     localparam CNT_WIDTH        = $clog2(N);
//     localparam SHIFT            = CNT_WIDTH;
//     localparam ROUND_BIAS       = (1 << (SHIFT-1));
//     localparam ACC_WIDTH        = SQUARED_WIDTH + CNT_WIDTH;
//     localparam EX2_WIDTH        = SQUARED_WIDTH;
//     localparam DOUT_WIDTH       = ACC_WIDTH + CNT_WIDTH;


//     /////////////// F/F /////////////////
//     reg         [1:0]               n_state,   c_state;
//     reg         [ACC_WIDTH-1:0]     n_acc,     c_acc;
//     reg         [CNT_WIDTH-1:0]     n_cnt_acc, c_cnt_acc;
//     reg signed  [EX2_WIDTH-1:0]     n_Ex2,     c_Ex2;


//     // ----------------------------------- Hybrid Square -----------------------------------
//     wire [SQUARED_WIDTH-1:0]    squared_x;
//     wire [3:0]                  H, L;
//     wire [7:0]                  H_sq, L_sq, HxL;
//     wire [7:0]                  x_abs;


//     // Separate the absolute value of an 8-bit input into upper (H) and lower (L) 4 bits
//     assign x_abs = i_x[7] ? (~i_x + 8'd1) : i_x;        // = (i_x < 0) ? -i_x : i_x
//     assign H = x_abs[7:4];
//     assign L = x_abs[3:0];


//     // 4bit square lut instance
//     Imp_mult4b_lut u_H_sq (.addr(H), .data(H_sq));
//     Imp_mult4b_lut u_L_sq (.addr(L), .data(L_sq));


//     // 4x4 mult
//     assign HxL = H * L;


//     // X^2 = 256*H^2 + 32*H*L + L^2
//     assign squared_x = (H_sq << 8) + (HxL << 5) + L_sq;
//     // ------------------------------------------------------------------------------------


//     // Rounding Logic
//     wire [ACC_WIDTH:0]      acc_rounded;
//     wire [EX2_WIDTH-1:0]    avg_ex2;        

//     assign acc_rounded = {1'b0, c_acc} + ROUND_BIAS;        // 1비트 여유
//     assign avg_ex2     = acc_rounded >> SHIFT;              // = floor((x+b)/2^k)


//     // F/F
//     always @(posedge i_clk or negedge i_rstn)
//     begin
//         if (!i_rstn)
//         begin
//             c_state   <= IDLE;
//             c_acc     <= 0;
//             c_cnt_acc <= 0;
//             c_Ex2     <= 0;
//         end
//         else
//         begin
//             c_state   <= n_state;
//             c_acc     <= n_acc;
//             c_cnt_acc <= n_cnt_acc;
//             c_Ex2     <= n_Ex2;
//         end
//     end


//     // n_state
//     always @(*)
//     begin
//         n_state = c_state;
//         case (c_state)
//             IDLE   : if (i_valid)                n_state = ACC;
//             ACC    : if (c_cnt_acc == N-1)       n_state = DIVIDE;
//             DIVIDE :                             n_state = DONE;          
//             DONE   :                             n_state = IDLE;
//         endcase
//     end
    
//     // n_cnt_acc
//     always @(*)
//     begin
//         n_cnt_acc = c_cnt_acc;
//         case (c_state)
//             IDLE :                  n_cnt_acc = 0;
//             ACC  : if (i_valid)     n_cnt_acc = c_cnt_acc + 1;
//         endcase
//     end

//     // n_acc
//     always @(*)
//     begin
//         n_acc = c_acc;
//         case (c_state)
//             IDLE :                  n_acc = 0;
//             ACC  : if (i_valid)     n_acc = c_acc + squared_x;
//         endcase
//     end

//     // n_Ex2
//     always @(*)
//     begin
//         n_Ex2 = c_Ex2;
//         case (c_state)
//             IDLE   :                  n_Ex2 = 0;
//             DIVIDE :                  n_Ex2 = avg_ex2;
//         endcase
//     end




//     // 최종 출력
//     assign o_Ex2      = (c_state == DONE) ? c_Ex2 : 0;
//     assign o_Ex2_done = (c_state == DONE);

// endmodule




// ----------------------------------------------------------------------------------
// Module: Imp_Ex2_Unit
// Description: 
//   - Calculates the expectation of x^2 for 128 int8 inputs.
//   - Processes 16 inputs in parallel over 8 clock cycles.
// ----------------------------------------------------------------------------------

module Imp_Ex2_Unit #(
    parameter N         = 128,
    parameter DATA_W    = 8,
    parameter EX2_W     = DATA_W * 2
)(
    input                           i_clk,
    input                           i_rstn,
    input                           i_valid,         
    input       [DATA_W*N-1:0]      i_x,    
    
    output      [EX2_W-1:0]         o_Ex2,  
    output                          o_Ex2_done        
);


    // FSM states
    localparam IDLE   = 2'd0;
    localparam ACC    = 2'd1;
    localparam DIVIDE = 2'd2;
    localparam DONE   = 2'd3;

    // Bit width
    localparam CHUNK_SIZE   = 16;                           //  Units to process at the same time
    localparam NUM_CHUNK    = N / CHUNK_SIZE;                       
    localparam CNT_WIDTH    = $clog2(NUM_CHUNK);
    localparam P_SUM_WIDTH  = EX2_W + $clog2(CHUNK_SIZE);   // partial sum width                            
    localparam ACC_WIDTH    = P_SUM_WIDTH + CNT_WIDTH;
    localparam SHIFT        = $clog2(N);
    localparam ROUND_BIAS   = (1 << (SHIFT - 1));          



    /////////////// F/F /////////////////
    reg         [1:0]               c_state, n_state;
    reg         [ACC_WIDTH-1:0]     c_acc, n_acc;
    reg         [CNT_WIDTH-1:0]     c_cnt_acc, n_cnt_acc;
    reg         [CNT_WIDTH-1:0]     c_cnt_at, n_cnt_at;     // count_adder_tree
    reg  signed [EX2_W-1:0]         c_Ex2, n_Ex2;


    /////////////// Input Selection and Hybrid Square Logic ///////////////
    wire [DATA_W-1:0]           selected_x[CHUNK_SIZE-1:0];     // Used for squared operations
    wire [EX2_W-1:0]            squared_x[CHUNK_SIZE-1:0];      // Squared input -> Used for adder tree 
    wire [P_SUM_WIDTH-1:0]      partial_sum;                    // Output of adder tree
    wire [3:0]                  H, L;
    wire [7:0]                  H_sq, L_sq, HxL;
    wire [7:0]                  x_abs;

    assign current_x = i_x[(c_cnt_acc * DATA_W) +: DATA_W];


    /////////////// Select a 16-element chunk from the input vector ///////////////
    genvar i;
    generate
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin : gen_input_mux
            assign selected_x[i] = i_x[(c_cnt_acc * CHUNK_SIZE * DATA_W) + (i * DATA_W) +: DATA_W];
        end
    endgenerate


    /////////////// Parallel Hybrid Square - Instantiate 16 Hybrid Square units ///////////////
    genvar j;
    generate
        for (j = 0; j < CHUNK_SIZE; j = j + 1) begin : gen_squared_x
            wire [3:0] H, L;
            wire [7:0] H_sq, L_sq, HxL;
            wire [7:0] x_abs;
            
            assign x_abs = selected_x[j][DATA_W-1] ? (~selected_x[j] + 1'b1) : selected_x[j];
            assign H = x_abs[7:4];
            assign L = x_abs[3:0];
            
            Imp_mult4b_lut u_H_sq (.addr(H), .data(H_sq));
            Imp_mult4b_lut u_L_sq (.addr(L), .data(L_sq));
            
            assign HxL = H * L;
            assign squared_x[j] = (H_sq << 8) + (HxL << 5) + L_sq;
        end
    endgenerate


    /////////////// Flatten squared_x -> Input of Adder tree ///////////////
    wire [EX2_W*CHUNK_SIZE-1:0]     at_i;

    genvar k;
    generate
        for (k = 0; k < CHUNK_SIZE; k = k + 1) begin : flatten_squared_x
            assign at_i[(k+1)*EX2_W-1 -: EX2_W] = squared_x[k];
        end
    endgenerate


    /////////////// Adder Tree instance ///////////////
    wire                            at_en;
    wire                            at_done;

    assign at_en = (c_state == ACC);

    adder_tree #(
        .DBW(EX2_W),
        .N(CHUNK_SIZE)
    ) u_adder_tree_16_input (
        .clk(i_clk),
        .arst_n(i_rstn),
        .wdata(at_i),
        .en(at_en),
        .rvalid(at_done),
        .rdata(partial_sum)
    );



    /////////////// Rounding and Division Logic ///////////////

    wire [ACC_WIDTH:0]  acc_rounded;
    wire [EX2_W-1:0]    avg_ex2;
    
    assign acc_rounded = {1'b0, c_acc} + ROUND_BIAS;
    assign avg_ex2     = acc_rounded >> SHIFT;


    // F/F
    always @(posedge i_clk or negedge i_rstn)
    begin
        if (!i_rstn)
        begin
            c_state   <= IDLE;
            c_acc     <= 0;
            c_cnt_acc <= 0;
            c_Ex2     <= 0;
            c_cnt_at  <= 0;
        end
        else
        begin
            c_state   <= n_state;
            c_acc     <= n_acc;
            c_cnt_acc <= n_cnt_acc;
            c_Ex2     <= n_Ex2;
            c_cnt_at  <= n_cnt_at;
        end
    end


    // n_state
    always @(*)
    begin
        n_state = c_state;
        case (c_state)
            IDLE   : if (i_valid)                                                   n_state = ACC;
            ACC    : if ((c_cnt_acc == NUM_CHUNK-1)&&(c_cnt_at == NUM_CHUNK-1))     n_state = DIVIDE;
            DIVIDE :                                                                n_state = DONE;
            DONE   :                                                                n_state = IDLE;
        endcase
    end
    
    // n_cnt_acc
    always @(*)
    begin
        n_cnt_acc = c_cnt_acc;
        case (c_state)
            IDLE  :                                 n_cnt_acc = 0;
            ACC   : if (c_cnt_acc < NUM_CHUNK-1)    n_cnt_acc = c_cnt_acc + 1'b1;
        endcase
    end


 // n_cnt_at
    always @(*)
    begin
        n_cnt_at = c_cnt_at;
        case (c_state)
            IDLE  :                                 n_cnt_at = 0;
            ACC   : if (at_done)                    n_cnt_at = c_cnt_at + 1'b1;
        endcase
    end


    // n_acc
    always @(*)
    begin
        n_acc = c_acc;
        case (c_state)
            IDLE  :                                 n_acc = 0;
            ACC   :  if (at_done && c_cnt_at > 0)   n_acc = c_acc + partial_sum;
        endcase
    end
    
    // n_Ex2
    always @(*)
    begin
        n_Ex2 = c_Ex2;
        case (c_state)
            IDLE   :                         n_Ex2 = 0;
            DIVIDE :                         n_Ex2 = avg_ex2;
        endcase
    end
    
    // Output
    assign o_Ex2      = (c_state == DONE) ? c_Ex2 : 0;
    assign o_Ex2_done = (c_state == DONE);

endmodule