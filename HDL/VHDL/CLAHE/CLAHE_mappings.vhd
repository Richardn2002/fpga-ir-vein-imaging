-- ****** I had trouble integrating the constant.vhd during simulation, so I hardcoded everything in port definition instead ****** 
LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY CLAHE_mappings IS
    PORT (
        clk : IN STD_LOGIC; -- clock input
        trg : IN STD_LOGIC; -- high for one period to start
        rdy : OUT STD_LOGIC; -- high for one period on completion

        img_in_en : OUT STD_LOGIC;
        img_in_addr : OUT STD_LOGIC_VECTOR(9 DOWNTO 0);
        img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        hist_mapping_inout_ren : OUT STD_LOGIC;
        hist_mapping_inout_wen : OUT STD_LOGIC;
        hist_mapping_inout_addr : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
        hist_mapping_inout_din : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        hist_mapping_inout_dout : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END CLAHE_mappings;

ARCHITECTURE arch OF CLAHE_mappings IS
    CONSTANT CLAHE_PATCH_X : NATURAL := 32;
    CONSTANT CLAHE_PATCH_Y : NATURAL := 32;
    CONSTANT CLAHE_CLIP_LIMIT : NATURAL := 10;
    CONSTANT PATCH_SIZE : NATURAL := CLAHE_PATCH_X * CLAHE_PATCH_Y;

    TYPE state_type IS (IDLE, INIT_HIST, READ_PIXEL, UPDATE_HIST, WAIT_HIST_READ,
        REDIST, CALC_MAPPING, WRITE_MAPPING, DONE);
    SIGNAL state, state_next : state_type;

    SIGNAL x_coord, x_coord_next : unsigned(4 DOWNTO 0); -- from 0 to 31
    SIGNAL y_coord, y_coord_next : unsigned(4 DOWNTO 0); -- from 0 to 31
    SIGNAL hist_idx, hist_idx_next : unsigned(7 DOWNTO 0); -- from 0 to 255

    SIGNAL pixel_value, pixel_value_next : unsigned(7 DOWNTO 0); -- from 0 to 255
    SIGNAL hist_value, hist_value_next : unsigned(7 DOWNTO 0);
    SIGNAL excess, excess_next : unsigned(15 DOWNTO 0);
    SIGNAL redist_val, redist_val_next : unsigned(7 DOWNTO 0);
    SIGNAL accumulator, accumulator_next : unsigned(31 DOWNTO 0); -- for cumulative distribution function
    SIGNAL s_normed : unsigned(31 DOWNTO 0); -- normalized, 32 bits

    SIGNAL wait_counter, wait_counter_next : unsigned(1 DOWNTO 0);

    -- Output signals
    SIGNAL rdy_next : STD_LOGIC;
    SIGNAL img_in_en_next : STD_LOGIC;
    SIGNAL img_in_addr_next : STD_LOGIC_VECTOR(9 DOWNTO 0);
    SIGNAL hist_mapping_inout_ren_next : STD_LOGIC;
    SIGNAL hist_mapping_inout_wen_next : STD_LOGIC;
    SIGNAL hist_mapping_inout_addr_next : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL hist_mapping_inout_dout_next : STD_LOGIC_VECTOR(7 DOWNTO 0);

BEGIN

    -- Sequential process: update state and registers
    seq_process : PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            state <= state_next;
            x_coord <= x_coord_next;
            y_coord <= y_coord_next;
            hist_idx <= hist_idx_next;
            pixel_value <= pixel_value_next;
            hist_value <= hist_value_next;
            excess <= excess_next;
            redist_val <= redist_val_next;
            accumulator <= accumulator_next;
            wait_counter <= wait_counter_next;

            rdy <= rdy_next;
            img_in_en <= img_in_en_next;
            img_in_addr <= img_in_addr_next;
            hist_mapping_inout_ren <= hist_mapping_inout_ren_next;
            hist_mapping_inout_wen <= hist_mapping_inout_wen_next;
            hist_mapping_inout_addr <= hist_mapping_inout_addr_next;
            hist_mapping_inout_dout <= hist_mapping_inout_dout_next;
        END IF;
    END PROCESS;

    -- Combinational process: compute next state and outputs
    comb_process : PROCESS (state, trg, img_in_d, hist_mapping_inout_din,
        x_coord, y_coord, hist_idx, pixel_value, hist_value,
        excess, redist_val, accumulator, wait_counter)
    BEGIN
        -- Default: hold current values
        state_next <= state;
        x_coord_next <= x_coord;
        y_coord_next <= y_coord;
        hist_idx_next <= hist_idx;
        pixel_value_next <= pixel_value;
        hist_value_next <= hist_value;
        excess_next <= excess;
        redist_val_next <= redist_val;
        accumulator_next <= accumulator;
        wait_counter_next <= wait_counter;

        rdy_next <= '0';
        img_in_en_next <= '0';
        img_in_addr_next <= (OTHERS => '0');
        hist_mapping_inout_ren_next <= '0';
        hist_mapping_inout_wen_next <= '0';
        hist_mapping_inout_addr_next <= (OTHERS => '0');
        hist_mapping_inout_dout_next <= (OTHERS => '0');

        -- s_normed = clip(s * 256 / PATCH_SIZE, 0, 255)
        -- s * 256 / 1024 = s / 4
        s_normed <= resize(shift_right(accumulator, 2), 32); -- divide by 4 (shift right 2 bits), concurrent w this process

        CASE state IS
            WHEN IDLE =>
                excess_next <= (OTHERS => '0');
                accumulator_next <= (OTHERS => '0'); -- accumulator is set to 0, it's used in the CALC_MAPPING state

                IF trg = '1' THEN
                    state_next <= INIT_HIST;
                    hist_idx_next <= (OTHERS => '0');
                END IF;

                -- Initialize histogram
            WHEN INIT_HIST =>
                hist_mapping_inout_wen_next <= '1'; -- start writing
                hist_mapping_inout_addr_next <= STD_LOGIC_VECTOR(hist_idx); -- hist_idx updated in the sequential process
                hist_mapping_inout_dout_next <= (OTHERS => '0'); -- no data exit

                IF hist_idx = 255 THEN
                    -- initialize histogram and x, y coordinates
                    hist_idx_next <= (OTHERS => '0');
                    x_coord_next <= (OTHERS => '0');
                    y_coord_next <= (OTHERS => '0');
                    state_next <= READ_PIXEL;
                    -- stop writing 
                    hist_mapping_inout_wen_next <= '0';
                ELSE
                    hist_idx_next <= hist_idx + 1;
                END IF;

                -- Read pixels
            WHEN READ_PIXEL =>
                img_in_en_next <= '1'; -- ready to read image
                img_in_addr_next <= STD_LOGIC_VECTOR(y_coord & x_coord); -- concatenate y and x coordinates to form a memory address
                wait_counter_next <= (OTHERS => '0');
                state_next <= UPDATE_HIST;

                -- Wait for pixel data and read current histogram value
            WHEN UPDATE_HIST =>
                IF wait_counter = 0 THEN
                    pixel_value_next <= unsigned(img_in_d);
                    hist_mapping_inout_ren_next <= '1';
                    hist_mapping_inout_addr_next <= img_in_d;
                    wait_counter_next <= wait_counter + 1;
                ELSE
                    hist_value_next <= unsigned(hist_mapping_inout_din); -- assume 1 clk cycle read latency
                    state_next <= WAIT_HIST_READ;
                END IF;

                -- Update histogram with clipping
            WHEN WAIT_HIST_READ =>
                hist_mapping_inout_wen_next <= '1';
                hist_mapping_inout_addr_next <= STD_LOGIC_VECTOR(pixel_value);

                IF hist_value = (CLAHE_CLIP_LIMIT - 2) THEN -- Richard used == instead of >= in the Python script, I keep it his way. 
                    excess_next <= excess + 1;
                    hist_mapping_inout_dout_next <= STD_LOGIC_VECTOR(hist_value);
                ELSE
                    hist_mapping_inout_dout_next <= STD_LOGIC_VECTOR(hist_value + 1);
                END IF;

                -- Move to next pixel
                IF x_coord = (CLAHE_PATCH_X - 1) THEN
                    x_coord_next <= (OTHERS => '0');
                    IF y_coord = (CLAHE_PATCH_Y - 1) THEN
                        -- Done reading all pixels
                        state_next <= REDIST;
                        hist_idx_next <= (OTHERS => '0'); -- set to zero so that in the redistribution process we start from 0th
                        redist_val_next <= excess(15 DOWNTO 8); -- divide by 256
                    ELSE
                        y_coord_next <= y_coord + 1;
                        state_next <= READ_PIXEL;
                    END IF;
                ELSE
                    x_coord_next <= x_coord + 1;
                    state_next <= READ_PIXEL;
                END IF;

                -- Redistribute excess
            WHEN REDIST =>
                hist_mapping_inout_ren_next <= '1';
                hist_mapping_inout_addr_next <= STD_LOGIC_VECTOR(hist_idx); -- ***** starts from 0, reset after WAIT_HIST_READ state
                wait_counter_next <= (OTHERS => '0');
                state_next <= CALC_MAPPING;

                -- Calculate CDF mapping
            WHEN CALC_MAPPING =>
                IF wait_counter = 0 THEN -- wait to fetch data from RAM
                    wait_counter_next <= wait_counter + 1;
                ELSE
                    hist_value_next <= unsigned(hist_mapping_inout_din) + redist_val; -- uniform addition, hist_mapping_inout_din is the histogram count read from current memory
                    accumulator_next <= accumulator + resize(unsigned(hist_mapping_inout_din), 32) + resize(redist_val, 32); -- cumulative sum(find the number of pixels with intensity below current bin index)
                    state_next <= WRITE_MAPPING;
                END IF;

                -- Write normalized CDF value
            WHEN WRITE_MAPPING =>
                hist_mapping_inout_wen_next <= '1'; -- mapping write is enabled  
                hist_mapping_inout_addr_next <= STD_LOGIC_VECTOR(hist_idx);

                ------- clip/extract ------- 
                IF s_normed > 255 THEN
                    hist_mapping_inout_dout_next <= x"FF"; -- clip to 255
                ELSE
                    hist_mapping_inout_dout_next <= STD_LOGIC_VECTOR(s_normed(7 DOWNTO 0));
                END IF;

                IF hist_idx = 255 THEN
                    state_next <= DONE;
                ELSE
                    -- wrote to memory, continue to read the next histogram bin
                    hist_idx_next <= hist_idx + 1; -- next cycle of the loop, we fetch from the next RAM address
                    state_next <= REDIST;
                END IF;

            WHEN DONE =>
                rdy_next <= '1';
                state_next <= IDLE;

        END CASE;
    END PROCESS;

END arch;

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