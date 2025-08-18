// -> FIFO IP 



module Input_Buffer #(
    parameter DEPTH = 8,        
    parameter WIDTH = 9	 // signed        
)(
    input                  i_clk,
    input                  i_rstn,
    input                  i_wr_en,       
    input      [WIDTH-1:0] i_wr_data,     
    input                  i_rd_en,    

    output reg [WIDTH-1:0] o_rd_data,     
    output                 o_full,        
    output                 o_empty        
);


    // Pointer bits
    localparam PTR_WIDTH = $clog2(DEPTH);
    
    // Memory Array
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    
    
    //////////////////// F/F ////////////////////
	reg [WIDTH-1:0]		c_temp, n_temp;

    reg [PTR_WIDTH-1:0] c_wr_ptr, n_wr_ptr; 	// Write Pointer
    reg [PTR_WIDTH-1:0] c_rd_ptr, n_rd_ptr;		// Read Pointer

    reg [PTR_WIDTH:0]   c_cnt, n_cnt;			// Counter
    

    // FIFO state
//    assign o_full  = (c_cnt == DEPTH);
    assign o_full  = (c_cnt == DEPTH-1);
	assign o_empty = (c_cnt == 0);
    
/*
	// Memory Initialization
    gnevar i;
    generate
        for (i = 0; i < DEPTH; i = i + 1) begin : mem_reset
            always @(posedge i_clk, negedge i_rstn) 
            begin
                if (!i_rstn)
                    mem[i] <= {WIDTH{1'b0}};
            end
        end
    endgenerate
*/

	
    // F/F
    always @(posedge i_clk, negedge i_rstn) 
    begin
        if (!i_rstn) 
        begin
        	c_temp	 	<= 0;
          	c_wr_ptr 	<= 0;
            c_rd_ptr 	<= 0;
            c_cnt  		<= 0;
        end

        else
        begin
        	c_temp		<= n_temp;
			c_wr_ptr 	<= n_wr_ptr;
            c_rd_ptr 	<= n_rd_ptr;
            c_cnt  		<= n_cnt;
        end
    end





	// Memory Write
	integer i;
	always @(posedge i_clk, negedge i_rstn)
	begin
		if (!i_rstn)
		begin
			for (i = 0; i < DEPTH; i = i + 1)
				mem[i]	<= 0;
		end

		else if (i_wr_en)
		begin
			if (c_wr_ptr == 0)	mem[c_wr_ptr]	<= c_temp;
			else				mem[c_wr_ptr]	<= i_wr_data;
		end
	end




	// n_temp
	always @(*)
	begin
		n_temp = c_temp;
		if (i_wr_en && c_wr_ptr == 0)		n_temp = i_wr_data;		// To prevent zero from being store where the input is empty
	end


	// n_wr_ptr
	always @(*)
	begin
		n_wr_ptr = c_wr_ptr;
		if (i_wr_en)						n_wr_ptr = c_wr_ptr + 1;	// Even if the 'o_full' is 1, one data will be read(out) 
//		if (i_wr_en && !o_full)				n_wr_ptr = c_wr_ptr + 1;
//		else if (o_full && i_wr_en)			n_wr_ptr = c_wr_ptr + 1;
		else if (!i_wr_en)					n_wr_ptr = {WIDTH{1'b0}};
	end


	// n_rd_ptr
	always @(*)
	begin
		n_rd_ptr = c_rd_ptr;
		if (i_rd_en && !o_empty)			n_rd_ptr = c_rd_ptr + 1;
	end

/*
	// n_cnt
	always @(*)
	begin
		n_cnt = c_cnt;
		if (i_wr_en && i_rd_en)				n_cnt = c_cnt;
		else if (i_wr_en && !o_full)		n_cnt = c_cnt + 1;
		else if (i_rd_en && !o_empty)		n_cnt = c_cnt - 1;
	end
*/

	always @(*)
	begin
    	n_cnt = c_cnt;
    	if 		(i_wr_en && !o_full)		n_cnt = c_cnt + 1;
    	else if (i_rd_en && !o_empty)		n_cnt = c_cnt - 1;
	end


	// Memory Array			mem storage order : 1, 2, 3, 4, 5, 6, 7, 0  
	always @(*)
    begin
//    	mem[c_wr_ptr] = i_wr_data;
//    	mem[c_wr_ptr] = c_temp;
//    	o_rd_data   = {WIDTH{1'b0}};
    	// Write
//        if (i_wr_en)			 					mem[c_wr_ptr] = i_wr_data;
// **        if (c_wr_ptr == 0)							mem[c_wr_ptr] = c_temp;
// **        else										mem[c_wr_ptr] = i_wr_data;
//        else if (!i_wr_en && c_wr_ptr == DEPTH-1)	mem[c_wr_ptr] = i_wr_data;
        // Read
		if (i_rd_en && !o_empty)					o_rd_data = mem[c_rd_ptr];
		else										o_rd_data   = {WIDTH{1'b0}};
    end
endmodule

