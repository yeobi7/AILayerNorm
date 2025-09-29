//==========================================================================
// Module: Imp_Ex_Unit
// Description: Squared and accumulated (Ex) N 8-bit inputs, and averaged.
// Divide Using shift  
//==========================================================================

module Imp_Ex_Unit #(
    parameter N         = 128,       // # of input
    parameter DATA_W    = 8                           
)(
    input                           i_clk,
    input                           i_rstn,
    input                           i_valid,
    input  signed [DATA_W*N-1:0]      i_x,        // -127 ~ 127

    output                          o_Ex_done,
    output signed [DATA_W-1:0]      o_Ex                  
);

    // FSM state
    localparam IDLE   = 2'd0;
    localparam ACC    = 2'd1;
    localparam DIVIDE = 2'd2; 
    localparam DONE   = 2'd3;

    // bit width
    
    localparam CHUNK_SIZE    = 16;
    localparam NUM_CHUNK     = N / CHUNK_SIZE;
    localparam CNT_WIDTH     = $clog2(NUM_CHUNK); 
    localparam P_SUM_WIDTH   = DATA_W + $clog2(CHUNK_SIZE); 
    localparam ACC_WIDTH     = P_SUM_WIDTH + CNT_WIDTH;    
    localparam SHIFT         = $clog2(N);
    localparam ROUND_BIAS    = (1 << (SHIFT - 1));              // rounding bias


    /////////////// F/F /////////////////
    reg         [1:0]                   n_state,        c_state;
    reg signed  [ACC_WIDTH-1:0]         n_acc,          c_acc;
    reg         [CNT_WIDTH-1:0]         n_cnt_acc,      c_cnt_acc;
    reg         [CNT_WIDTH-1:0]         n_cnt_at,       c_cnt_at;     // count_adder_tree
    reg signed  [7:0]                   n_Ex,           c_Ex;
    // reg signed  [P_SUM_WIDTH-1:0]       n_partial_sum,  c_partial_sum;


    /////////////// Wire ///////////////
    wire signed [DATA_W-1:0]            selected_x[CHUNK_SIZE-1:0];
    wire        [DATA_W*CHUNK_SIZE-1:0] at_i;
    wire signed [P_SUM_WIDTH-1:0]       partial_sum;
    wire                                at_en;          // adder tree enable
    wire                                at_done;        // adder tree done




    /////////////// Select a 16-element chunk from the input vector ///////////////
    genvar i;
    generate
        for (i = 0; i < CHUNK_SIZE; i = i + 1) begin : gen_mux
            assign selected_x[i] = i_x[(c_cnt_acc * CHUNK_SIZE * DATA_W) + (i * DATA_W) +: DATA_W];
        end
    endgenerate


    /////////////// Flatten input -> Input of Adder tre ///////////////
    genvar j;
    generate
        for (j = 0; j < CHUNK_SIZE; j = j + 1) begin : gen_flatten
            assign at_i[(j+1)*DATA_W-1 -: DATA_W] = selected_x[j];
        end
    endgenerate


    // adder_tree module instance

    assign at_en = (c_state == ACC);

    adder_tree #(
        .DBW(DATA_W),
        .N(CHUNK_SIZE)
    ) u_adder_tree(
        .clk(i_clk),
        .arst_n(i_rstn),
        .wdata(at_i),
        .en(at_en),  
        .rvalid(at_done),
        .rdata(partial_sum)
        // .rdata(c_partial_sum)
    );


    // Rounding 
    wire        [ACC_WIDTH:0]   acc_rounded;
    wire signed [DATA_W-1:0]    avg_ex;

    assign acc_rounded = $signed(c_acc) + ROUND_BIAS;
    assign avg_ex      = acc_rounded >> SHIFT;




    // F/F
    always @(posedge i_clk or negedge i_rstn)
    begin
        if (!i_rstn)
        begin
            c_state         <= IDLE;
            c_acc           <= 0;
            c_cnt_acc       <= 0;
            c_cnt_at        <= 0;
            c_Ex            <= 0;
            // c_partial_sum   <= 0;
        end
        else
        begin
            c_state         <= n_state;
            c_acc           <= n_acc;
            c_cnt_at        <= n_cnt_at;
            c_cnt_acc       <= n_cnt_acc;
            c_Ex            <= n_Ex;
            // c_partial_sum   <= n_partial_sum;
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
            IDLE :                                  n_cnt_acc = 0;
            ACC  : if (c_cnt_acc < NUM_CHUNK-1)     n_cnt_acc = c_cnt_acc + 1;
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
            IDLE :                                  n_acc = 0;
            ACC  : if (at_done && c_cnt_at > 0)     n_acc = c_acc + partial_sum;
        endcase
    end



    // n_Ex
    always @(*)
    begin
        n_Ex = c_Ex;
        case (c_state)
            IDLE   :                n_Ex = 0;
            DIVIDE :                n_Ex = avg_ex;
        endcase
    end


    // // n_partial_sum
    // always @(*)
    // begin
    //     n_partial_sum = c_partial_sum;
    //     case (c_state)
    //         IDLE   :                n_partial_sum = 0;
    //         DIVIDE :                n_partial_sum = ;
    //     endcase
    // end



    // output
    assign o_Ex      = (c_state == DONE) ? c_Ex : 0;
    assign o_Ex_done = (c_state == DONE);

endmodule