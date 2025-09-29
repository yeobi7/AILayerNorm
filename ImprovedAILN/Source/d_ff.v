`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2022/02/28 12:21:09
// Design Name: 
// Module Name: D_FF
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module d_ff(clk,arst_n,d,q);

parameter DBW=1;
input clk,arst_n;
input [DBW-1:0] d;
output reg [DBW-1:0]   q;

always @ (posedge clk or negedge arst_n) begin
  
  if(!arst_n)  q<=0;
  else         q<=d;

end

endmodule



