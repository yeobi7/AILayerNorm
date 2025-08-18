module Ex2_Unit(
    input       	i_clk,         	
    input       	i_rstn,       	
    input       	i_valid,       	
       	
    input 	[8:0]   i_x,      	
    input	[1:0]   i_alpha,   	 // PTF scailing ��	
    input 	[7:0]   i_inv_n,	 // 1/C, Q0.8 -> LUT


    output  		o_Ex2_done,
    output  [31:0]  o_Ex2        // Ex2 ���
);

	parameter	IDLE = 2'd0;
	parameter	ACC  = 2'd1;
	parameter	DONE = 2'd2;


	////////////// F/F ///////////////
	reg [ 1:0] n_state,   	c_state;
    reg [19:0] n_acc, 	  	c_acc;   		// 16 bits + log2(N) = 16 + 4 -> 20 bits
    reg [ 2:0] n_cnt_acc, 	c_cnt_acc;
    reg [31:0] n_Ex2, 	  	c_Ex2;
    reg [15:0] n_shifted_x, c_shifted_x;

	////////////// reg //////////////
	reg			shift_signal;  	 // Dynamic Compression signal (1�̸� << 4, 0�̸� << 2)

    reg	[3:0]  	compressed_x;
    reg	[7:0]   square_x;

//    reg [15:0] acc;
//	reg	[2:0]  acc_cnt;


	////////////// wire ////////////

	wire [ 3:0] dyn_shift;	
    wire [ 3:0] total_shift;		// max shift(decompress) : 4(shift_signal = 1) + 4(alpha = 2) 
//    wire [15:0] shifted_x;		// squared_x1(8bits) << max shift(8bits) -> 16bits


//    wire [39:0] mult_ex2;		// acc(20bits) * i_inv_n(20bits) -> 40bits
//    wire [31:0] mean_square;



	// F/F
	always @(posedge i_clk or negedge i_rstn) 
    begin
        if (!i_rstn)
        begin
        	c_state	  	<= 2'd0;
            c_acc 	  	<= 20'd0;
            c_cnt_acc 	<= 3'd0;
            c_Ex2	  	<= 32'd0;
            c_shifted_x <= 16'd0;
        end

        else 
        begin 
        	c_state   	<= n_state;
        	c_acc	  	<= n_acc;
        	c_cnt_acc 	<= n_cnt_acc;
        	c_Ex2		<= n_Ex2;
        	c_shifted_x <= n_shifted_x;
        end
    end


	// n_state
	always @(*)
	begin
		n_state = c_state;
		case (c_state)
		IDLE : if (i_valid)			 	n_state = ACC;
		ACC  : if (c_cnt_acc == 3'd7)	n_state = DONE;
		DONE : 							n_state = IDLE;
		endcase
	end


	// n_cnt_acc
	always @(*)
	begin
		n_cnt_acc = c_cnt_acc;
		case (c_state)
		IDLE :							n_cnt_acc = 3'd0;
		ACC  :							n_cnt_acc = c_cnt_acc + 1;
		endcase
	end


	// n_acc
	always @(*)
	begin
		n_acc = c_acc;
		case (c_state)
		IDLE : 							n_acc = 20'd0;
		ACC  :							n_acc = c_acc + c_shifted_x;
		endcase
	end


	// n_Ex
	always @(*)
	begin
		n_Ex2 = c_Ex2;
		case (c_state)
		IDLE : 							n_Ex2 = 32'd0;
		ACC  : if (c_cnt_acc == 3'd7)	n_Ex2 = ((c_acc + c_shifted_x) * {12'b0, i_inv_n}) >>> 8;
		endcase
	end


	
	// n_shifted_x
	always @(*)
	begin
		n_shifted_x = c_shifted_x;
		case (c_state)
		IDLE : 							n_shifted_x = 16'd0;
		ACC  : 							n_shifted_x = square_x << total_shift;
		endcase
	end


    // Dynamic Compression : i_x 8bits -> 4bits
    always @(*) 
    begin
        if (i_x[7:6] != 2'b00)  // right shift : 4
        begin
            if (i_x[7:3] == 5'd31)		// overflow ����
                compressed_x = 4'd15;
            else
            	if (i_x[3] == 1'b1)	// Rounding
                	compressed_x = (i_x[7:3] >> 1) + 1; 
                else
                	compressed_x = i_x[7:3] >> 1;
            shift_signal = 1'b1;	
        end 
        
        else 
        begin
        	if (i_x[1] == 1'b1)	// Rounding
        		compressed_x = i_x[5:2] + 1;
        	else 
        		compressed_x = i_x[5:2];
            shift_signal = 1'b0;
        end
    end



    // 4bits square LUT : 4bits compressed_x�� 8bits square_x
    always @(*) 
    begin
        case (compressed_x)
            4'd0:  square_x= 8'd0;
            4'd1:  square_x= 8'd1;
            4'd2:  square_x= 8'd4;
            4'd3:  square_x= 8'd9;
            4'd4:  square_x= 8'd16;
            4'd5:  square_x= 8'd25;
            4'd6:  square_x= 8'd36;
            4'd7:  square_x= 8'd49;
            4'd8:  square_x= 8'd64;
            4'd9:  square_x= 8'd81;
            4'd10: square_x= 8'd100;
            4'd11: square_x= 8'd121;
            4'd12: square_x= 8'd144;
            4'd13: square_x= 8'd169;
            4'd14: square_x= 8'd196;
            4'd15: square_x= 8'd225;
            default: square_x = 8'd0;
        endcase
	end



    // Decompression 
    assign dyn_shift = (shift_signal) ? 4 : 0;
    assign total_shift = dyn_shift + (i_alpha << 1);  // total_shift = (4 or 0) + 2*i_alpha

//    assign shifted_x = square_x << total_shift;


	

    // Accumulator
    /*
    always @(posedge i_clk or negedge i_rstn) 
    begin
        if (!i_rstn)
        begin
            acc 	<= 16'd0;
            acc_cnt <= 3'd0;
        end
        
        else if (i_valid) 
        begin
            if (16'hFFFF - acc < shifted_x)
                acc <= 16'hFFFF;
            else
                acc <= acc + shifted_x;
            acc_cnt <= acc_cnt + 1;
        end
    end
	*/


	// Accumulator * inv_n (1/C)
//    assign mult_ex2    = c_acc * {12'b0, i_inv_n};
//    assign mean_square = mult_ex2 >>> 8;	// Q0.8 -> INT8

    assign o_Ex2 = (o_Ex2_done) ? c_Ex2 : 32'd0;		// 8bit

    assign o_Ex2_done = (c_state == DONE);
endmodule

