//==========================================================================
// Module: Imp_cal_sqrt
// Description: Calculates the precise square root of a 16-bit integer
//==========================================================================

module Imp_cal_sqrt (
    input            i_clk,
    input            i_rstn,
    input            i_start,
    input    [15:0]  i_data,
    output   [7:0]   o_sqrt,
    output           o_done
);

    // --- FSM State ---
    localparam IDLE   = 2'd0;
    localparam Stage1 = 2'd1; // Norm, LU, DeNorm
    localparam Stage2 = 2'd2; // Rounding
    localparam DONE   = 2'd3;


    // --- F/F ---
    reg [1:0]   c_state, n_state;
    reg [15:0]  c_temp_i,       n_temp_i;      // Holds original input data for rounding
    reg [7:0]   c_temp_sqrt,    n_temp_sqrt;   // Holds the floored sqrt value
    reg [7:0]   c_temp_o,       n_temp_o;
    //reg         c_done,         n_done;


    // --- Register ---
    reg  [4:0]  msb_pos; 
    wire [2:0]  k;
    wire [15:0] d_norm;
    wire [7:0]  lut_addr;
    wire [7:0]  sqrt_lut_o;
    wire [7:0]  quotient;        
    wire [15:0] quotient_sq;
    wire [15:0] boundary;       // using in rounding



    // --- find MSB position ---
    always @(*) 
    begin
        if      (i_data[15]) msb_pos = 15;
        else if (i_data[14]) msb_pos = 14;
        else if (i_data[13]) msb_pos = 13;
        else if (i_data[12]) msb_pos = 12;
        else if (i_data[11]) msb_pos = 11;
        else if (i_data[10]) msb_pos = 10;
        else if (i_data[9])  msb_pos = 9;
        else if (i_data[8])  msb_pos = 8;
        else if (i_data[7])  msb_pos = 7;
        else if (i_data[6])  msb_pos = 6;
        else if (i_data[5])  msb_pos = 5;
        else if (i_data[4])  msb_pos = 4;
        else if (i_data[3])  msb_pos = 3;
        else if (i_data[2])  msb_pos = 2;
        else if (i_data[1])  msb_pos = 1;
        else                 msb_pos = 0;
    end


    // --- sqrt LUT Instance ---
    Imp_sqrt_lut U_sqrt (
        .addr(lut_addr),
        .data(sqrt_lut_o)
    );


    assign k = msb_pos >> 1;
    assign d_norm = i_data << ((7 - k) << 1);
    assign lut_addr = d_norm[15:8];
    assign quotient = sqrt_lut_o >> (7 - k);
    

    // --- Rounding logic ---
    assign quotient_sq = c_temp_sqrt * c_temp_sqrt;
    assign boundary = quotient_sq + c_temp_sqrt;


    // --- Sequential Logic ---
    always @(posedge i_clk or negedge i_rstn) 
    begin
        if (!i_rstn) 
        begin
            c_state      <= IDLE;
            c_temp_i     <= 16'd0;
            c_temp_sqrt  <= 8'd0;
            c_temp_o     <= 8'd0;
            //c_done       <= 1'd0;
        end 
        else 
        begin
            c_state      <= n_state;
            c_temp_i     <= n_temp_i;
            c_temp_sqrt  <= n_temp_sqrt;        
            c_temp_o     <= n_temp_o;
            //c_done       <= n_done;
        end
    end


    // --- Combinational Logic ---

    // n_state
    always @(*) 
    begin
        n_state = c_state;
        case (c_state)
            IDLE    :   if (i_start)            n_state = Stage1;
            Stage1  :                           n_state = Stage2;
            Stage2  :                           n_state = DONE;
            DONE    :                           n_state = IDLE;
        endcase
    end


    // n_temp_i
    always @(*)
    begin
        n_temp_i = c_temp_i;
        case (c_state)
            IDLE    :   if (i_start)            n_temp_i = i_data;
        endcase
    end


    // n_temp_sqrt
    always@(*)
    begin
        n_temp_sqrt = c_temp_sqrt;
        case (c_state)
            Stage1  :                           n_temp_sqrt = quotient;
        endcase
    end


    // n_temp_o
    always@(*)
    begin
        n_temp_o = c_temp_o;
        case (c_state)
            Stage2  : 
            begin
                if      (c_temp_i == 0)         n_temp_o = 8'b0;
                else if (c_temp_i > boundary)   n_temp_o = c_temp_sqrt + 1;
                else                            n_temp_o = c_temp_sqrt;    
            end
        endcase  
    end

    /*
    // n_done
    always@(*)
    begin
        n_done = 1'b0;
        case (c_state)
            DONE    :                           n_done = 1'b1;
        endcase
    end
    */


    // Output
    assign o_done = (c_state == DONE);
    assign o_sqrt = (c_state == DONE) ? c_temp_o : 8'd0;

endmodule