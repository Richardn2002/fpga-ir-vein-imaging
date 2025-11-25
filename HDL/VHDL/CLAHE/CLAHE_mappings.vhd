LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

USE work.constants.ALL;

ENTITY CLAHE_mappings IS
    PORT (
        clk : IN STD_LOGIC; -- clock input
        trg : IN STD_LOGIC; -- high for one period to start
        rdy : OUT STD_LOGIC; -- high for one period on completion

        img_in_en : OUT STD_LOGIC;
        img_in_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_PATCH_ADDR_BITS - 1 DOWNTO 0);
        img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        hist_mapping_inout_ren : OUT STD_LOGIC;
        hist_mapping_inout_wen : OUT STD_LOGIC;
        hist_mapping_inout_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS - 1 DOWNTO 0);
        hist_mapping_inout_din : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        hist_mapping_inout_dout : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
end CLAHE_mappings;

architecture of CLAHE_mappings is
    constant CLAHE_PATCH_X : natural := 32;
    constant CLAHE_PATCH_Y : natural := 32;
    constant CLAHE_CLIP_LIMIT : natural := 10;
    constant PATCH_SIZE : natural := CLAHE_PATCH_X * CLAHE_PATCH_Y;

    type state_type IS (IDLE, INIT_HIST, READ_PIXEL, UPDATE_HIST, WAIT_HIST_READ, 
                        REDIST, CALC_MAPPING, WRITE_MAPPING, DONE);
    signal state, state_next : state_type;

    signal x_coord, x_coord_next : unsigned(4 DOWNTO 0); -- from 0 to 31
    signal y_coord, y_coord_next : unsigned(4 DOWNTO 0); -- from 0 to 31
    signal hist_idx, hist_idx_next : unsigned(7 DOWNTO 0); -- from 0 to 255
    
    signal pixel_value, pixel_value_next : unsigned(7 DOWNTO 0); -- from 0 to 255
    signal hist_value, hist_value_next : unsigned(7 DOWNTO 0);
    signal excess, excess_next : unsigned(15 DOWNTO 0);
    signal redist, redist_next : unsigned(7 DOWNTO 0);
    signal accumulator, accumulator_next : unsigned(31 DOWNTO 0); -- for cumulative distribution function
    signal s_normed : unsigned(31 DOWNTO 0); -- normalized, 32 bits
    
    signal wait_counter, wait_counter_next : unsigned(1 DOWNTO 0);

    -- Output signals
    signal rdy_next : STD_LOGIC;
    signal img_in_en_next : STD_LOGIC;
    signal img_in_addr_next : STD_LOGIC_VECTOR(constants.CLAHE_PATCH_ADDR_BITS - 1 DOWNTO 0);
    signal hist_mapping_inout_ren_next : STD_LOGIC;
    signal hist_mapping_inout_wen_next : STD_LOGIC;
    signal hist_mapping_inout_addr_next : STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS - 1 DOWNTO 0);
    signal hist_mapping_inout_dout_next : STD_LOGIC_VECTOR(7 DOWNTO 0);

begin

    -- Sequential process: update state and registers
    seq_process : process(clk)
    begin
        if rising_edge(clk) THEN
            state <= state_next;
            x_coord <= x_coord_next;
            y_coord <= y_coord_next;
            hist_idx <= hist_idx_next;
            pixel_value <= pixel_value_next;
            hist_value <= hist_value_next;
            excess <= excess_next;
            redist <= redist_next;
            accumulator <= accumulator_next;
            wait_counter <= wait_counter_next;
            
            rdy <= rdy_next;
            img_in_en <= img_in_en_next;
            img_in_addr <= img_in_addr_next;
            hist_mapping_inout_ren <= hist_mapping_inout_ren_next;
            hist_mapping_inout_wen <= hist_mapping_inout_wen_next;
            hist_mapping_inout_addr <= hist_mapping_inout_addr_next;
            hist_mapping_inout_dout <= hist_mapping_inout_dout_next;
        end if;
    end process;

    -- Combinational process: compute next state and outputs
    comb_process : process(state, trg, img_in_d, hist_mapping_inout_din,
                           x_coord, y_coord, hist_idx, pixel_value, hist_value,
                           excess, redist, accumulator, wait_counter)
    begin
        -- Default: hold current values
        state_next <= state;
        x_coord_next <= x_coord;
        y_coord_next <= y_coord;
        hist_idx_next <= hist_idx;
        pixel_value_next <= pixel_value;
        hist_value_next <= hist_value;
        excess_next <= excess;
        redist_next <= redist;
        accumulator_next <= accumulator;
        wait_counter_next <= wait_counter;
        
        rdy_next <= '0';
        img_in_en_next <= '0';
        img_in_addr_next <= (others => '0');
        hist_mapping_inout_ren_next <= '0';
        hist_mapping_inout_wen_next <= '0';
        hist_mapping_inout_addr_next <= (others => '0');
        hist_mapping_inout_dout_next <= (others => '0');
        
        -- s_normed = clip(s * 256 / PATCH_SIZE, 0, 255)
        -- s * 256 / 1024 = s / 4
        s_normed <= resize(shift_right(accumulator, 2), 32); -- divide by 4 (shift right 2 bits), concurrent w this process

        case state IS
            when IDLE =>
                excess_next <= (others => '0');
                accumulator_next <= (others => '0');    -- accumulator is set to 0, it's used in the CALC_MAPPING state
                
                if trg = '1' THEN
                    state_next <= INIT_HIST;
                    hist_idx_next <= (others => '0');
                end if;

            -- Initialize histogram
            when INIT_HIST =>
                hist_mapping_inout_wen_next <= '1';     -- start writing
                hist_mapping_inout_addr_next <= std_logic_vector(hist_idx);     -- hist_idx updated in the sequential process
                hist_mapping_inout_dout_next <= (others => '0');    -- no data exit
                
                if hist_idx = 255 THEN
                    -- initialize histogram and x, y coordinates
                    hist_idx_next <= (others => '0');
                    x_coord_next <= (others => '0');
                    y_coord_next <= (others => '0');
                    state_next <= READ_PIXEL;
                    -- stop writing 
                    hist_mapping_inout_wen_next <= '0';
                ELSE
                    hist_idx_next <= hist_idx + 1;
                end if;

            -- Read pixels
            when READ_PIXEL =>
                img_in_en_next <= '1';      -- ready to read image
                img_in_addr_next <= std_logic_vector(y_coord & x_coord);    -- concatenate y and x coordinates to form a memory address
                wait_counter_next <= (others => '0');
                state_next <= UPDATE_HIST;

            -- Wait for pixel data and read current histogram value
            when UPDATE_HIST =>
                if wait_counter = 0 THEN
                    pixel_value_next <= unsigned(img_in_d);
                    hist_mapping_inout_ren_next <= '1';
                    hist_mapping_inout_addr_next <= img_in_d;
                    wait_counter_next <= wait_counter + 1;
                ELSE
                    hist_value_next <= unsigned(hist_mapping_inout_din);    -- assume 1 clk cycle read latency
                    state_next <= WAIT_HIST_READ;
                end if;

            -- Update histogram with clipping
            when WAIT_HIST_READ =>
                hist_mapping_inout_wen_next <= '1';
                hist_mapping_inout_addr_next <= std_logic_vector(pixel_value);
                
                if hist_value = (CLAHE_CLIP_LIMIT - 2) THEN     -- Richard used == instead of >= in the Python script, I keep it his way. 
                    excess_next <= excess + 1;
                    hist_mapping_inout_dout_next <= std_logic_vector(hist_value);
                ELSE
                    hist_mapping_inout_dout_next <= std_logic_vector(hist_value)+1;
                end if;
                
                -- Move to next pixel
                if x_coord = (CLAHE_PATCH_X - 1) THEN
                    x_coord_next <= (others => '0');
                    if y_coord = (CLAHE_PATCH_Y - 1) THEN
                        -- Done reading all pixels
                        state_next <= REDIST;
                        hist_idx_next <= (others => '0');   -- set to zero so that in the redistribution process we start from 0th
                        redist_next <= excess(15 DOWNTO 8); -- divide by 256
                    ELSE
                        y_coord_next <= y_coord + 1;
                        state_next <= READ_PIXEL;
                    end if;
                ELSE
                    x_coord_next <= x_coord + 1;
                    state_next <= READ_PIXEL;
                end if;

            -- Redistribute excess
            when REDIST =>
                hist_mapping_inout_ren_next <= '1';
                hist_mapping_inout_addr_next <= std_logic_vector(hist_idx); -- ***** starts from 0, reset after WAIT_HIST_READ state
                wait_counter_next <= (others => '0');
                state_next <= CALC_MAPPING;

            -- Calculate CDF mapping
            when CALC_MAPPING =>
                if wait_counter = 0 THEN    -- wait to fetch data from RAM
                    wait_counter_next <= wait_counter + 1;
                ELSE
                    hist_value_next <= unsigned(hist_mapping_inout_din) + redist;   -- uniform addition, hist_mapping_inout_din is the histogram count read from current memory
                    accumulator_next <= accumulator + resize(unsigned(hist_mapping_inout_din), 32) + resize(redist, 32);    -- cumulative sum(find the number of pixels with intensity below current bin index)
                    state_next <= WRITE_MAPPING;
                end if;

            -- Write normalized CDF value
            when WRITE_MAPPING =>
                hist_mapping_inout_wen_next <= '1';   -- mapping write is enabled  
                
                ------- clip/extract ------- 
                if s_normed > 255 THEN
                    hist_mapping_inout_dout_next <= x"FF";  -- clip to 255
                ELSE
                    hist_mapping_inout_dout_next <= std_logic_vector(s_normed(7 DOWNTO 0));
                end if;
                
                if hist_idx = 255 THEN
                    state_next <= DONE;
                ELSE
                    -- wrote to memory, continue to read the next histogram bin
                    hist_idx_next <= hist_idx + 1;  -- next cycle of the loop, we fetch from the next RAM address
                    state_next <= REDIST;
                end if;

            when DONE =>
                rdy_next <= '1';
                state_next <= IDLE;

        end case;
    end process;

end behavioral;



-- LIBRARY IEEE;
-- USE IEEE.std_logic_1164.ALL;

-- USE work.constants;

-- ENTITY CLAHE_mappings IS
--     PORT (
--         clk : IN STD_LOGIC; -- clock input
--         trg : IN STD_LOGIC; -- high for one period to start
--         rdy : OUT STD_LOGIC; -- high for one period on completion

--         img_in_en : OUT STD_LOGIC;
--         img_in_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_PATCH_ADDR_BITS - 1 DOWNTO 0);
--         img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

--         hist_mapping_inout_ren : OUT STD_LOGIC;
--         hist_mapping_inout_wen : OUT STD_LOGIC;
--         hist_mapping_inout_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS - 1 DOWNTO 0);
--         hist_mapping_inout_din : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
--         hist_mapping_inout_dout : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
--     );
-- END CLAHE_mappings;
