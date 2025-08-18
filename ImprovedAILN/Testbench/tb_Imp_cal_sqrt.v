module tb_Imp_cal_sqrt();
    reg          i_clk;
    reg          i_rstn;
    reg          i_start;
    reg  [15:0]  i_data;

    wire         o_done;
    wire [7:0]   o_sqrt;


always #5 i_clk = ~i_clk;

 Imp_cal_sqrt u_cal_sqrt (
    .i_clk(i_clk),
    .i_rstn(i_rstn),
    .i_start(i_start),
    .i_data(i_data),
    .o_done(o_done),
    .o_sqrt(o_sqrt)
);

initial 
begin
    i_clk = 1'd0;   i_rstn = 1'd0;  i_start = 1'd0;
    #1 i_rstn = 1'd1;   #1 i_rstn = 1'd0;   #2 i_rstn = 1'd1;

    #5 i_start = 1'd1;

    #10  i_data = 16'd4000;
    #100 i_data = 16'd1000;
    #100 i_data = 16'd40000;      
    #100 i_data = 16'd100;
    #100 i_data = 16'd4;
    #100 i_data = 16'd5326;
    #100 i_data = 16'd11094;
end

endmodule