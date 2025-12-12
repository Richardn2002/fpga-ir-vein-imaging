LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

ENTITY CLAHE_mappings IS
    GENERIC (
        PATCH_IDX : NATURAL := 0 -- Which patch this instance handles (0-15)
    );
    PORT (
        clk : IN STD_LOGIC;
        trg : IN STD_LOGIC;
        rdy : OUT STD_LOGIC;

        -- Image RAM Interface
        img_in_en : OUT STD_LOGIC;
        img_in_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0); -- Extended for 128*128
        img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        -- Histogram/Mapping RAM Interface
        hist_mapping_inout_ren : OUT STD_LOGIC;
        hist_mapping_inout_wen : OUT STD_LOGIC;
        hist_mapping_inout_addr : OUT STD_LOGIC_VECTOR(11 DOWNTO 0); -- 12 bits for 4096 addresses
        hist_mapping_inout_din : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        hist_mapping_inout_dout : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END CLAHE_mappings;

ARCHITECTURE arch OF CLAHE_mappings IS
    CONSTANT CLAHE_PATCH_X : NATURAL := 32;
    CONSTANT CLAHE_PATCH_Y : NATURAL := 32;
    CONSTANT CLAHE_CLIP_LIMIT : NATURAL := 10;
    CONSTANT INPUT_WIDTH : NATURAL := 128;

    -- Calculate patch position from PATCH_IDX
    CONSTANT PATCH_X_OFFSET : NATURAL := (PATCH_IDX MOD 4) * 32; -- 0, 32, 64, 96
    CONSTANT PATCH_Y_OFFSET : NATURAL := (PATCH_IDX / 4) * 32; -- 0, 32, 64, 96
    CONSTANT PATCH_BASE_ADDR : NATURAL := PATCH_IDX * 256; -- Base address in mapping RAM

    TYPE state_type IS (
        IDLE,
        INIT_HIST_WRITE,
        READ_PIXEL_ADDR,
        READ_PIXEL_WAIT,
        READ_HIST_ADDR,
        READ_HIST_WAIT,
        UPDATE_HIST_CHECK,
        UPDATE_HIST_WRITE,
        CHECK_LOOP,
        REDIST_READ_ADDR,
        REDIST_READ_WAIT,
        CALC_MAPPING,
        WRITE_MAPPING,
        DONE
    );
    SIGNAL state, state_next : state_type;

    SIGNAL x_coord, x_coord_next : unsigned(4 DOWNTO 0);
    SIGNAL y_coord, y_coord_next : unsigned(4 DOWNTO 0);
    SIGNAL hist_idx, hist_idx_next : unsigned(7 DOWNTO 0);

    SIGNAL pixel_value, pixel_value_next : unsigned(7 DOWNTO 0);
    SIGNAL hist_value, hist_value_next : unsigned(7 DOWNTO 0);
    SIGNAL excess, excess_next : unsigned(15 DOWNTO 0);
    SIGNAL redist_val, redist_val_next : unsigned(7 DOWNTO 0);

    SIGNAL accumulator, accumulator_next : unsigned(31 DOWNTO 0);
    SIGNAL s_normed : unsigned(31 DOWNTO 0);

    -- Output Buffers
    SIGNAL rdy_next : STD_LOGIC;
    SIGNAL img_in_en_next : STD_LOGIC;
    SIGNAL img_in_addr_next : STD_LOGIC_VECTOR(13 DOWNTO 0);
    SIGNAL hist_mapping_inout_ren_next : STD_LOGIC;
    SIGNAL hist_mapping_inout_wen_next : STD_LOGIC;
    SIGNAL hist_mapping_inout_addr_next : STD_LOGIC_VECTOR(11 DOWNTO 0);
    SIGNAL hist_mapping_inout_dout_next : STD_LOGIC_VECTOR(7 DOWNTO 0);

BEGIN

    -- Math: s_normed = s * 256 / 1024  =>  s / 4
    s_normed <= resize(shift_right(accumulator, 2), 32);

    ----------------------------------------------------------------------------
    -- 1. Sequential Process: Registers & Sync Outputs
    ----------------------------------------------------------------------------
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

            -- Update Outputs on Clock Edge
            rdy <= rdy_next;
            img_in_en <= img_in_en_next;
            img_in_addr <= img_in_addr_next;
            hist_mapping_inout_ren <= hist_mapping_inout_ren_next;
            hist_mapping_inout_wen <= hist_mapping_inout_wen_next;
            hist_mapping_inout_addr <= hist_mapping_inout_addr_next;
            hist_mapping_inout_dout <= hist_mapping_inout_dout_next;
        END IF;
    END PROCESS;

    ----------------------------------------------------------------------------
    -- 2. Combinational Process: Logic
    ----------------------------------------------------------------------------
    comb_process : PROCESS (state, trg, img_in_d, hist_mapping_inout_din,
        x_coord, y_coord, hist_idx, pixel_value, hist_value,
        excess, redist_val, accumulator, s_normed)
        VARIABLE img_addr : INTEGER;
        VARIABLE hist_addr : INTEGER;
    BEGIN
        -- Defaults
        state_next <= state;
        x_coord_next <= x_coord;
        y_coord_next <= y_coord;
        hist_idx_next <= hist_idx;
        pixel_value_next <= pixel_value;
        hist_value_next <= hist_value;
        excess_next <= excess;
        redist_val_next <= redist_val;
        accumulator_next <= accumulator;

        rdy_next <= '0';
        img_in_en_next <= '0';
        img_in_addr_next <= (OTHERS => '0');
        hist_mapping_inout_ren_next <= '0';
        hist_mapping_inout_wen_next <= '0';
        hist_mapping_inout_addr_next <= (OTHERS => '0');
        hist_mapping_inout_dout_next <= (OTHERS => '0');

        CASE state IS
            WHEN IDLE =>
                excess_next <= (OTHERS => '0');
                accumulator_next <= (OTHERS => '0');
                IF trg = '1' THEN
                    state_next <= INIT_HIST_WRITE;
                    hist_idx_next <= (OTHERS => '0');
                END IF;

                --------------------------------------------------------------------
                -- PHASE 1: Initialize Histogram to 0
                --------------------------------------------------------------------
            WHEN INIT_HIST_WRITE =>
                hist_mapping_inout_wen_next <= '1';
                hist_addr := PATCH_BASE_ADDR + to_integer(hist_idx);
                hist_mapping_inout_addr_next <= STD_LOGIC_VECTOR(to_unsigned(hist_addr, 12));
                hist_mapping_inout_dout_next <= (OTHERS => '0');

                IF hist_idx = 255 THEN
                    hist_idx_next <= (OTHERS => '0');
                    x_coord_next <= (OTHERS => '0');
                    y_coord_next <= (OTHERS => '0');
                    state_next <= READ_PIXEL_ADDR;
                ELSE
                    hist_idx_next <= hist_idx + 1;
                END IF;

                --------------------------------------------------------------------
                -- PHASE 2: Build Histogram
                --------------------------------------------------------------------
            WHEN READ_PIXEL_ADDR =>
                img_in_en_next <= '1';
                -- Calculate absolute address: (PATCH_Y_OFFSET + y_coord) * 128 + (PATCH_X_OFFSET + x_coord)
                img_addr := (PATCH_Y_OFFSET + to_integer(y_coord)) * INPUT_WIDTH +
                    (PATCH_X_OFFSET + to_integer(x_coord));
                img_in_addr_next <= STD_LOGIC_VECTOR(to_unsigned(img_addr, 14));
                state_next <= READ_PIXEL_WAIT;

            WHEN READ_PIXEL_WAIT =>
                -- Wait 1 cycle for img_in_d to become valid
                state_next <= READ_HIST_ADDR;

            WHEN READ_HIST_ADDR =>
                -- img_in_d is valid here
                pixel_value_next <= unsigned(img_in_d);

                -- Read existing count from Histogram RAM
                hist_mapping_inout_ren_next <= '1';
                hist_addr := PATCH_BASE_ADDR + to_integer(unsigned(img_in_d));
                hist_mapping_inout_addr_next <= STD_LOGIC_VECTOR(to_unsigned(hist_addr, 12));
                state_next <= READ_HIST_WAIT;

            WHEN READ_HIST_WAIT =>
                -- Wait 1 cycle for hist_mapping_inout_din to become valid
                state_next <= UPDATE_HIST_CHECK;

            WHEN UPDATE_HIST_CHECK =>
                hist_value_next <= unsigned(hist_mapping_inout_din);

                -- Strict Python Match: "if hist[pixel] == limit - 2: excess += 1"
                IF unsigned(hist_mapping_inout_din) = (CLAHE_CLIP_LIMIT - 2) THEN
                    excess_next <= excess + 1;
                END IF;
                state_next <= UPDATE_HIST_WRITE;

            WHEN UPDATE_HIST_WRITE =>
                hist_mapping_inout_wen_next <= '1';
                hist_addr := PATCH_BASE_ADDR + to_integer(pixel_value);
                hist_mapping_inout_addr_next <= STD_LOGIC_VECTOR(to_unsigned(hist_addr, 12));

                -- Strict Python Match: "else: hist[pixel] += 1"
                IF hist_value = (CLAHE_CLIP_LIMIT - 2) THEN
                    hist_mapping_inout_dout_next <= STD_LOGIC_VECTOR(hist_value); -- Do not increment
                ELSE
                    hist_mapping_inout_dout_next <= STD_LOGIC_VECTOR(hist_value + 1); -- Increment
                END IF;
                state_next <= CHECK_LOOP;

            WHEN CHECK_LOOP =>
                IF x_coord = (CLAHE_PATCH_X - 1) THEN
                    x_coord_next <= (OTHERS => '0');
                    IF y_coord = (CLAHE_PATCH_Y - 1) THEN
                        -- Done Reading
                        state_next <= REDIST_READ_ADDR;
                        hist_idx_next <= (OTHERS => '0');

                        -- Strict Python Match: "redist = excess // 256"
                        redist_val_next <= excess(15 DOWNTO 8);
                    ELSE
                        y_coord_next <= y_coord + 1;
                        state_next <= READ_PIXEL_ADDR;
                    END IF;
                ELSE
                    x_coord_next <= x_coord + 1;
                    state_next <= READ_PIXEL_ADDR;
                END IF;

                --------------------------------------------------------------------
                -- PHASE 3: Redistribution & CDF
                --------------------------------------------------------------------
            WHEN REDIST_READ_ADDR =>
                hist_mapping_inout_ren_next <= '1';
                hist_addr := PATCH_BASE_ADDR + to_integer(hist_idx);
                hist_mapping_inout_addr_next <= STD_LOGIC_VECTOR(to_unsigned(hist_addr, 12));
                state_next <= REDIST_READ_WAIT;

            WHEN REDIST_READ_WAIT =>
                state_next <= CALC_MAPPING;

            WHEN CALC_MAPPING =>
                -- Python: "hist[i] += redist" AND "s += hist[i]"
                -- Update accumulator with current bin + redistribution
                accumulator_next <= accumulator + resize(unsigned(hist_mapping_inout_din), 32) + resize(redist_val, 32);
                state_next <= WRITE_MAPPING;

            WHEN WRITE_MAPPING =>
                hist_mapping_inout_wen_next <= '1';
                hist_addr := PATCH_BASE_ADDR + to_integer(hist_idx);
                hist_mapping_inout_addr_next <= STD_LOGIC_VECTOR(to_unsigned(hist_addr, 12));

                -- Write normalized value (calculated concurrently)
                IF s_normed > 255 THEN
                    hist_mapping_inout_dout_next <= x"FF";
                ELSE
                    hist_mapping_inout_dout_next <= STD_LOGIC_VECTOR(s_normed(7 DOWNTO 0));
                END IF;

                IF hist_idx = 255 THEN
                    state_next <= DONE;
                ELSE
                    hist_idx_next <= hist_idx + 1;
                    state_next <= REDIST_READ_ADDR;
                END IF;

            WHEN DONE =>
                rdy_next <= '1';
                IF trg = '0' THEN
                    state_next <= IDLE;
                END IF;

            WHEN OTHERS =>
                state_next <= IDLE;
        END CASE;
    END PROCESS;

END arch;