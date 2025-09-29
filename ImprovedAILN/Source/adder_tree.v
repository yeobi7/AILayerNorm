`timescale 1ns / 1ps
//////////////////////////NODE///////////////////////////////////////////////////////
// Company: syd
// Engineer: jmu
// 
// Create Date: 2021/06/22 21:24:53
// Design Name: 
// Module Name: adder_tree
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


      
module adder_tree#(
         parameter DBW =8 ,  
         parameter N =32)  
    (    
      clk,
      arst_n,
      wdata,
      en,
      rvalid,
      rdata );     
    //`include "clog2_function.vh"
   localparam STAGES_NUM = $clog2(N); 

  // (* gated_clock = "true" *) input    clk;
    input    clk;
    input    arst_n;
    input   signed [DBW*N -1 :0] wdata;
    input    en;
    output   rvalid;
    output   signed[DBW+STAGES_NUM-1:0] rdata;
  


  

  generate






    if (N==128) begin :N_128
      reduction_adder_tree_128#(
      .DBW(DBW),  
      .N(N))  
      reduction_adder_tree_128(    
      .clk(clk),
      .arst_n(arst_n),
      .wdata(wdata),
      .en(en),
      .rvalid(rvalid),
      .rdata(rdata) );     

    end   if (N==64) begin :N_64
      reduction_adder_tree_64#(
      .DBW(DBW),  
      .N(N))  
      reduction_adder_tree_64(    
      .clk(clk),
      .arst_n(arst_n),
      .wdata(wdata),
      .en(en),
      .rvalid(rvalid),
      .rdata(rdata) );     

    end   if (N==32) begin :N_32
      reduction_adder_tree_32#(
      .DBW(DBW),  
      .N(N))  
      reduction_adder_tree_32(    
      .clk(clk),
      .arst_n(arst_n),
      .wdata(wdata),
      .en(en),
      .rvalid(rvalid),
      .rdata(rdata) );     

    end   if (N==16) begin:N_16
      reduction_adder_tree_16#(
      .DBW(DBW),  
      .N(N))  
      reduction_adder_tree_16(    
      .clk(clk),
      .arst_n(arst_n),
      .wdata(wdata),
      .en(en),
      .rvalid(rvalid),
      .rdata(rdata) );     

    end else if (N==9 )begin : N_9
        reduction_adder_tree_9#(
          .DBW(DBW),  
          .N(N))  
        reduction_adder_tree_9(    
          .clk(clk),
          .arst_n(arst_n),
          .wdata(wdata),
          .en(en),
          .rvalid(rvalid),
          .rdata(rdata) );     
    end else if (N==24 )begin : N_24
        reduction_adder_tree_24#(
          .DBW(DBW),  
          .N(N))  
        reduction_adder_tree_24(    
          .clk(clk),
          .arst_n(arst_n),
          .wdata(wdata),
          .en(en),
          .rvalid(rvalid),
          .rdata(rdata) );     
    end else if (N==48)begin :N_48
        reduction_adder_tree_48#(
          .DBW(DBW),  
          .N(N))  
        reduction_adder_tree_48(    
          .clk(clk),
          .arst_n(arst_n),
          .wdata(wdata),
          .en(en),
          .rvalid(rvalid),
          .rdata(rdata) );     
    end 

  endgenerate
endmodule




module reduction_adder_tree_128#(
         parameter DBW =8 ,  
         parameter N =128)  
    (    
      clk,
      arst_n,
      wdata,
      en,
      rvalid,
      rdata );     

      
      //`include "clog2_function.vh"
      // (* gated_clock = "true" *) input    clk;
      input    clk;
      input    arst_n;
      input    signed [DBW*N -1 :0] wdata;
      input    en;
      output   rvalid;
      output   reg signed [DBW+$clog2(N)-1:0] rdata;
      localparam STAGES_NUM = $clog2(N)+1; //tree level //input registering

      localparam N0=64;
      localparam N1=32;
      localparam N2=16;
      localparam N3=8;
      localparam N4=4;
      localparam N5=2;


      reg  signed [DBW*N -1 :0] wdata_reg;

      wire [STAGES_NUM:0] pipeline_en;
      assign pipeline_en[0]=en;
      assign rvalid =pipeline_en[STAGES_NUM];

        reg  signed [DBW+1-1:0] psum0[0:N0-1];
        reg  signed [DBW+2-1:0] psum1[0:N1-1];
        reg  signed [DBW+3-1:0] psum2[0:N2-1];
        reg  signed [DBW+4-1:0] psum3[0:N3-1];
        reg  signed [DBW+5-1:0] psum4[0:N4-1];
        reg  signed [DBW+6-1:0] psum5[0:N5-1];

      genvar j;
      generate
        for (j=0 ; j<STAGES_NUM; j=j+1) begin :pipe_ligne_status_reg
              d_ff pipeline_register (.clk(clk),.arst_n(arst_n),.d(pipeline_en[j]),.q(pipeline_en[j+1]));          
        end
      endgenerate

  integer i;

    always @ (posedge clk ) begin
    
        
          wdata_reg<=en ? wdata :wdata_reg; 
          
          for(i=0 ; i < N0; i=i+1) begin
                psum0[i]<=  $signed(wdata_reg[DBW*(i*2+0)+:DBW]) + $signed(wdata_reg[DBW*(i*2+1)+:DBW]);
          end
          for(i=0 ; i < N1; i=i+1) begin
                psum1[i]<=  $signed(psum0[(i*2+0)]) + $signed(psum0[(i*2+1)]);
          end

          for(i=0 ; i < N2; i=i+1) begin
                psum2[i]<=  $signed(psum1[(i*2+0)]) + $signed(psum1[(i*2+1)]);
          end
          
          for(i=0 ; i < N3; i=i+1) begin
                psum3[i]<=  $signed(psum2[(i*2+0)]) + $signed(psum2[(i*2+1)]);
          end
          
          for(i=0 ; i < N4; i=i+1) begin
                psum4[i]<=  $signed(psum3[(i*2+0)]) + $signed(psum3[(i*2+1)]);
          end
          
          for(i=0 ; i < N5; i=i+1) begin
                psum5[i]<=  $signed(psum4[(i*2+0)]) + $signed(psum4[(i*2+1)]);
          end
          
             rdata<=  $signed(psum5[1]) +  $signed(psum5[0]);
    end
    

endmodule




module reduction_adder_tree_64#(
         parameter DBW =8 ,  
         parameter N =64)  
    (    
      clk,
      arst_n,
      wdata,
      en,
      rvalid,
      rdata );     

      
      //`include "clog2_function.vh"
      // (* gated_clock = "true" *) input    clk;
      input    clk;
      input    arst_n;
      input    signed [DBW*N -1 :0] wdata;
      input    en;
      output   rvalid;
      output   reg signed [DBW+$clog2(N)-1:0] rdata;
      localparam STAGES_NUM = $clog2(N)+1; //tree level
      
      reg  signed [DBW*N -1 :0] wdata_reg;


      localparam N1=32;
      localparam N2=16;
      localparam N3=8;
      localparam N4=4;
      localparam N5=2;


      wire [STAGES_NUM:0] pipeline_en;
      assign pipeline_en[0]=en;
      assign rvalid =pipeline_en[STAGES_NUM];


        reg  signed [DBW+1-1:0] psum1[0:N1-1];
        reg  signed [DBW+2-1:0] psum2[0:N2-1];
        reg  signed [DBW+3-1:0] psum3[0:N3-1];
        reg  signed [DBW+4-1:0] psum4[0:N4-1];
        reg  signed [DBW+5-1:0] psum5[0:N5-1];

      genvar j;
      generate
        for (j=0 ; j<STAGES_NUM; j=j+1) begin:pipe_ligne_status_reg
              d_ff pipeline_register (.clk(clk),.arst_n(arst_n),.d(pipeline_en[j]),.q(pipeline_en[j+1]));          
        end
      endgenerate

  integer i;

    always @ (posedge clk ) begin
    
          wdata_reg<=en ? wdata :wdata_reg; 

          for(i=0 ; i < N1; i=i+1) begin
                psum1[i]<=  $signed(wdata_reg[DBW*(i*2+0)+:DBW]) + $signed(wdata_reg[DBW*(i*2+1)+:DBW]);
          end

          for(i=0 ; i < N2; i=i+1) begin
                psum2[i]<=  $signed(psum1[(i*2+0)]) + $signed(psum1[(i*2+1)]);
          end
          
          for(i=0 ; i < N3; i=i+1) begin
                psum3[i]<=  $signed(psum2[(i*2+0)]) + $signed(psum2[(i*2+1)]);
          end
          
          for(i=0 ; i < N4; i=i+1) begin
                psum4[i]<=  $signed(psum3[(i*2+0)]) + $signed(psum3[(i*2+1)]);
          end
          
          for(i=0 ; i < N5; i=i+1) begin
                psum5[i]<=  $signed(psum4[(i*2+0)]) + $signed(psum4[(i*2+1)]);
          end
          
             rdata<=  $signed(psum5[1]) +  $signed(psum5[0]);
        end

endmodule





module reduction_adder_tree_32#(
         parameter DBW =8 ,  
         parameter N =32)  
    (    
      clk,
      arst_n,
      wdata,
      en,
      rvalid,
      rdata );     

      
      //`include "clog2_function.vh"
      // (* gated_clock = "true" *) input    clk;
      input    clk;
      input    arst_n;
      input    signed [DBW*N -1 :0] wdata;
      input    en;
      output   rvalid;
      output   reg signed [DBW+$clog2(N)-1:0] rdata;
      localparam STAGES_NUM = $clog2(N)+1; //tree level

      reg  signed [DBW*N -1 :0] wdata_reg;
      localparam N2=16;
      localparam N3=8;
      localparam N4=4;
      localparam N5=2;


      wire [STAGES_NUM:0] pipeline_en;
      assign pipeline_en[0]=en;
      assign rvalid =pipeline_en[STAGES_NUM];


        reg  signed [DBW+1-1:0] psum2[0:N2-1];
        reg  signed [DBW+2-1:0] psum3[0:N3-1];
        reg  signed [DBW+3-1:0] psum4[0:N4-1];
        reg  signed [DBW+4-1:0] psum5[0:N5-1];

      genvar j;
      generate
        for (j=0 ; j<STAGES_NUM; j=j+1) begin:pipe_ligne_status_reg
              d_ff pipeline_register (.clk(clk),.arst_n(arst_n),.d(pipeline_en[j]),.q(pipeline_en[j+1]));          
        end
      endgenerate

  integer i;

    always @ (posedge clk ) begin
     
          wdata_reg <= en ? wdata: wdata_reg;
          for(i=0 ; i < N2; i=i+1) begin
                psum2[i]<=  $signed(wdata_reg[DBW*(i*2+0)+:DBW]) + $signed(wdata_reg[DBW*(i*2+1)+:DBW]);
          end
          
          for(i=0 ; i < N3; i=i+1) begin
                psum3[i]<=  $signed(psum2[(i*2+0)]) + $signed(psum2[(i*2+1)]);
          end
          
          for(i=0 ; i < N4; i=i+1) begin
                psum4[i]<=  $signed(psum3[(i*2+0)]) + $signed(psum3[(i*2+1)]);
          end
          
          for(i=0 ; i < N5; i=i+1) begin
                psum5[i]<=  $signed(psum4[(i*2+0)]) + $signed(psum4[(i*2+1)]);
          end
          
             rdata<=  $signed(psum5[1]) +  $signed(psum5[0]);
        end
    

endmodule


module reduction_adder_tree_16#(
         parameter DBW =8 ,  
         parameter N =16)  
    (    
      clk,
      arst_n,
      wdata,
      en,
      rvalid,
      rdata );     

      
      //`include "clog2_function.vh"
      // (* gated_clock = "true" *) input    clk;
      input    clk;
      input    arst_n;
      input    signed [DBW*N -1 :0] wdata;
      input    en;
      output   rvalid;
      output   reg signed [DBW+$clog2(N)-1:0] rdata;
      localparam STAGES_NUM = $clog2(N)+1; //tree level

      reg  signed [DBW*N -1 :0] wdata_reg;
      localparam N3=8;
      localparam N4=4;
      localparam N5=2;


      wire [STAGES_NUM:0] pipeline_en;
      assign pipeline_en[0]=en;
      assign rvalid =pipeline_en[STAGES_NUM];


        reg  signed [DBW+1-1:0] psum3[0:N3-1];
        reg  signed [DBW+2-1:0] psum4[0:N4-1];
        reg  signed [DBW+3-1:0] psum5[0:N5-1];

      genvar j;
      generate
        for (j=0 ; j<STAGES_NUM; j=j+1) begin:pipe_ligne_status_reg
              d_ff pipeline_register (.clk(clk),.arst_n(arst_n),.d(pipeline_en[j]),.q(pipeline_en[j+1]));          
        end
      endgenerate

  integer i;

    always @ (posedge clk ) begin

      wdata_reg<=en ? wdata : wdata_reg;

      for(i=0 ; i < N3; i=i+1) begin
            psum3[i]<=  $signed(wdata_reg[DBW*(i*2+0)+:DBW]) + $signed(wdata_reg[DBW*(i*2+1)+:DBW]);
      end
      
      for(i=0 ; i < N4; i=i+1) begin
            psum4[i]<=  $signed(psum3[(i*2+0)]) + $signed(psum3[(i*2+1)]);
      end
      
      for(i=0 ; i < N5; i=i+1) begin
            psum5[i]<=  $signed(psum4[(i*2+0)]) + $signed(psum4[(i*2+1)]);
      end
      
          rdata<=  $signed(psum5[1]) +  $signed(psum5[0]);
  end
  

endmodule




      
module reduction_adder_tree_9#(
         parameter DBW =8 ,  
         parameter N =9)  
    (    
      clk,
      arst_n,
      wdata,
      en,
      rvalid,
      rdata );     

      
      //`include "clog2_function.vh"
      // (* gated_clock = "true" *) input    clk;
      input    clk;
      input    arst_n;
      input    signed [DBW*N -1 :0] wdata;
      input    en;
      output   rvalid;
      output   reg signed [DBW+$clog2(N)-1:0] rdata;
      localparam STAGES_NUM = $clog2(N)+1; //tree level
      reg  signed [DBW*N -1 :0] wdata_reg;


      localparam N1=5;
      localparam N2=3;
      localparam N3=2;



      wire [STAGES_NUM:0] pipeline_en;
      assign pipeline_en[0]=en;
      assign rvalid =pipeline_en[STAGES_NUM];
      reg  signed [DBW+1-1:0] psum0[N1-1:0];
      reg  signed [DBW+2-1:0] psum1[N2-1:0];
      reg  signed [DBW+3-1:0] psum2[N3-1:0];

      genvar j;
      generate
        for (j=0 ; j<STAGES_NUM; j=j+1) begin:pipe_ligne_status_reg
              d_ff pipeline_register (.clk(clk),.arst_n(arst_n),.d(pipeline_en[j]),.q(pipeline_en[j+1]));          
        end
      endgenerate

  integer i;

    always @ (posedge clk ) begin
    
          
          wdata_reg <= en? wdata: wdata_reg;
          for(i=0 ; i < N1; i=i+1) begin
              if(i==N1-1)       psum0[i]<=  $signed(wdata_reg[DBW*i*2+:DBW]);
              else              psum0[i]<=  $signed(wdata_reg[DBW*(i*2+0)+:DBW]) + $signed(wdata_reg[DBW*(i*2+1)+:DBW]);
          end

          for(i=0 ; i < N2; i=i+1) begin
              if(i==N2-1)       psum1[i]<=  $signed(psum0[i*2]);
              else              psum1[i]<= $signed( psum0[i*2+0]) + $signed(psum0[i*2+1]);
                           
          end
          
          for(i=0 ; i < N3; i=i+1) begin
              if(i==N3-1)       psum2[i]<=  $signed(psum1[i*2]);
              else              psum2[i]<=  $signed(psum1[i*2+0]) + $signed(psum1[i*2+1]);
                     
         end 

         rdata<=  $signed(psum2[1]) +  $signed(psum2[0]);
    
    end

endmodule


      
module reduction_adder_tree_24#(
         parameter DBW =8 ,  
         parameter N =24)  
    (    
      clk,
      arst_n,
      wdata,
      en,
      rvalid,
      rdata );     

      
      //`include "clog2_function.vh"
      // (* gated_clock = "true" *) input    clk;
      input    clk;
      input    arst_n;
      input    signed [DBW*N -1 :0] wdata;
      input    en;
      output   rvalid;
      output   reg signed [DBW+$clog2(N)-1:0] rdata;
      localparam STAGES_NUM = $clog2(N)+1; //tree level
      reg  signed [DBW*N -1 :0] wdata_reg;

      localparam N1=12;
      localparam N2=6;
      localparam N3=3;
      localparam N4=2;



      wire [STAGES_NUM:0] pipeline_en;
      assign pipeline_en[0]=en;
      assign rvalid =pipeline_en[STAGES_NUM];
      reg  signed [DBW+1-1:0] psum1[N1-1:0];
      reg  signed [DBW+2-1:0] psum2[N2-1:0];
      reg  signed [DBW+3-1:0] psum3[N3-1:0];
      reg  signed [DBW+4-1:0] psum4[N4-1:0];

      genvar j;
      generate
        for (j=0 ; j<STAGES_NUM; j=j+1) begin:pipe_ligne_status_reg
              d_ff pipeline_register (.clk(clk),.arst_n(arst_n),.d(pipeline_en[j]),.q(pipeline_en[j+1]));          
        end
      endgenerate

  integer i;

    always @ (posedge clk ) begin
    
          wdata_reg<= en ? wdata : wdata_reg;
          for(i=0 ; i < N1; i=i+1) begin
              psum1[i]<=  $signed(wdata_reg[DBW*(i*2+0)+:DBW]) + $signed(wdata_reg[DBW*(i*2+1)+:DBW]);
          end

          for(i=0 ; i < N2; i=i+1) begin
              psum2[i]<= $signed( psum1[i*2+0]) + $signed(psum1[i*2+1]);    
          end
          
          for(i=0 ; i < N3; i=i+1) begin
              psum3[i]<= $signed( psum2[i*2+0]) + $signed(psum2[i*2+1]);    
          end
          
          for(i=0 ; i < N4; i=i+1) begin
              if(i==N4-1)       psum4[i]<=  $signed(psum3[i*2]);
              else              psum4[i]<=  $signed(psum3[i*2+0]) + $signed(psum3[i*2+1]);
         end 

         rdata<=  $signed(psum4[1]) +  $signed(psum4[0]);
  
    end

endmodule


      
module reduction_adder_tree_48#(
         parameter DBW =8 ,  
         parameter N =48)  
    (    
      clk,
      arst_n,
      wdata,
      en,
      rvalid,
      rdata );     

      
      //`include "clog2_function.vh"
      // (* gated_clock = "true" *) input    clk;
      input    clk;
      input    arst_n;
      input    signed [DBW*N -1 :0] wdata;
      input    en;
      output   rvalid;
      output   reg signed [DBW+$clog2(N)-1:0] rdata;
      localparam STAGES_NUM = $clog2(N)+1; //tree level
      reg  signed [DBW*N -1 :0] wdata_reg;
      localparam N0=24;
      localparam N1=12;
      localparam N2=6;
      localparam N3=3;
      localparam N4=2;



      wire [STAGES_NUM:0] pipeline_en;
      assign pipeline_en[0]=en;
      assign rvalid =pipeline_en[STAGES_NUM];

      reg  signed [DBW+1-1:0] psum0[N0-1:0];
      reg  signed [DBW+2-1:0] psum1[N1-1:0];
      reg  signed [DBW+3-1:0] psum2[N2-1:0];
      reg  signed [DBW+4-1:0] psum3[N3-1:0];
      reg  signed [DBW+5-1:0] psum4[N4-1:0];

      genvar j;
      generate
        for (j=0 ; j<STAGES_NUM; j=j+1) begin:pipe_ligne_status_reg
              d_ff pipeline_register (.clk(clk),.arst_n(arst_n),.d(pipeline_en[j]),.q(pipeline_en[j+1]));          
        end
      endgenerate

  integer i;

    always @ (posedge clk ) begin
          wdata_reg<= en? wdata : wdata_reg;
          for(i=0 ; i < N0; i=i+1) begin
              psum0[i]<=  $signed(wdata_reg[DBW*(i*2+0)+:DBW]) + $signed(wdata_reg[DBW*(i*2+1)+:DBW]);
          end

          for(i=0 ; i < N1; i=i+1) begin
              psum1[i]<= $signed( psum0[i*2+0]) + $signed(psum0[i*2+1]);    
          end

          for(i=0 ; i < N2; i=i+1) begin
              psum2[i]<= $signed( psum1[i*2+0]) + $signed(psum1[i*2+1]);    
          end
          
          for(i=0 ; i < N3; i=i+1) begin
              psum3[i]<= $signed( psum2[i*2+0]) + $signed(psum2[i*2+1]);    
          end
          
          for(i=0 ; i < N4; i=i+1) begin
              if(i==N4-1)       psum4[i]<=  $signed(psum3[i*2]);
              else              psum4[i]<=  $signed(psum3[i*2+0]) + $signed(psum3[i*2+1]);
         end 

         rdata<=  $signed(psum4[1]) +  $signed(psum4[0]);
    end

endmodule

