//==========================================================================
// Module: Imp_Affine_Unit
// Description: Calculates the var and std, Normalization with alpha, beta 
//==========================================================================

module Imp_Affine_Unit #(
    parameter N = 8
)(
    input                   i_clk,
    input                   i_rstn,
    input                   i_valid,
    input  signed   [7:0]   i_x,
    input  signed   [8:0]   i_Ex,
    input           [15:0]  i_Ex2,
    input  signed   [7:0]   i_alpha,
    input  signed   [7:0]   i_beta,

    output                  o_Affine_done,
    output signed   [7:0]   o_Affine         
);





endmodule