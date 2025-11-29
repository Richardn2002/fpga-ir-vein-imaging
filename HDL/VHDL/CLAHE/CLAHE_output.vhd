-- LIBRARY IEEE;
-- USE IEEE.std_logic_1164.ALL;

-- USE work.constants;

-- ENTITY CLAHE_output IS
--     PORT (
--         clk : IN STD_LOGIC; -- clock input
--         trg : IN STD_LOGIC; -- high for one period to start
--         rdy : OUT STD_LOGIC; -- high for one period on completion

--         img_in_en : OUT STD_LOGIC;
--         img_in_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_IMG_ADDR_BITS - 1 DOWNTO 0);
--         img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

--         mapping_in_en : OUT STD_LOGIC;
--         mapping_in_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS - 1 DOWNTO 0);
--         mapping_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

--         clahe_out_en : OUT STD_LOGIC;
--         clahe_out_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
--         clahe_out_d : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
--     );
-- END CLAHE_output;


-- Jack's notes:
-- This module is going to interface with an 'image RAM block' that stores the unmodified image and a 'CLAHE mappings' RAM block
-- For a 4x4 tile, 32x32 pixels per tile setup, the CLAHE mappings RAM block should store 4096 bytes of data
-- Its address is organized as tile_idx*256, where tile_idx = row*4 + column. So tiles 0-3 are the top row of tiles, tiles 4-7 are the second row, etc
-- And since each tile's LUT is a 0-255 mapping that's where the 256 comes from. So for example, tile 5 is row 1, column 1's tile and takes up addresses 1280-1535

-- In short, the CLAHE output will take in all of the mapping LUTs and the original image, and output the bilinearly interpolated and newly mapped image (CLAHE output)
-- The way the interpolation and sweep is implemented makes it so that a 128x128 turns into a 96x96 (the inner 96x96 pixels)

LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

USE work.constants;

ENTITY CLAHE_output IS
    PORT (
        clk  : IN  STD_LOGIC; -- clock input
        trg  : IN  STD_LOGIC; -- high for one period to start
        rdy  : OUT STD_LOGIC; -- high for one period on completion

        -- Input image RAM interface
        img_in_en   : OUT STD_LOGIC;
        img_in_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_IMG_ADDR_BITS - 1 DOWNTO 0);
        img_in_d    : IN  STD_LOGIC_VECTOR(7 DOWNTO 0);

        -- Mapping RAM interface (flattened 4x4x256 LUTs)
        mapping_in_en   : OUT STD_LOGIC;
        mapping_in_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS - 1 DOWNTO 0);
        mapping_in_d    : IN  STD_LOGIC_VECTOR(7 DOWNTO 0);

        -- CLAHE output RAM interface (96x96)
        clahe_out_en   : OUT STD_LOGIC;
        clahe_out_addr : OUT STD_LOGIC_VECTOR(constants.CLAHE_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        clahe_out_d    : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END CLAHE_output;


ARCHITECTURE rtl OF CLAHE_output IS

    CONSTANT INPUT_X  : integer := constants.INPUT_X;          -- 128
    CONSTANT INPUT_Y  : integer := constants.INPUT_Y;          -- 128;

    CONSTANT PATCH_X  : integer := constants.CLAHE_PATCH_X;    -- 32
    CONSTANT PATCH_Y  : integer := constants.CLAHE_PATCH_Y;    -- 32
    CONSTANT NUM_X    : integer := constants.CLAHE_PATCH_X_NUM;-- 4
    CONSTANT NUM_Y    : integer := constants.CLAHE_PATCH_Y_NUM;-- 4

    CONSTANT OUT_X    : integer := INPUT_X - PATCH_X;          -- 96
    CONSTANT OUT_Y    : integer := INPUT_Y - PATCH_Y;          -- 96

    CONSTANT TILE_LUT_SIZE : integer := 256;                   -- 256 gray levels
    TYPE state_t IS (
        IDLE,
        PIX_REQ,     
        PIX_WAIT,     
        TL_REQ, TL_WAIT,
        TR_REQ, TR_WAIT,
        BL_REQ, BL_WAIT,
        BR_REQ, BR_WAIT,
        COMPUTE,
        WRITE_OUT,
        DONE
    );

    SIGNAL state, state_next : state_t;

    SIGNAL x, x_next : integer range 0 TO OUT_X - 1;
    SIGNAL y, y_next : integer range 0 TO OUT_Y - 1;

    SIGNAL pixel_center, pixel_center_next : unsigned(7 DOWNTO 0);
    SIGNAL tl, tl_next : unsigned(7 DOWNTO 0);
    SIGNAL tr, tr_next : unsigned(7 DOWNTO 0);
    SIGNAL bl, bl_next : unsigned(7 DOWNTO 0);
    SIGNAL br, br_next : unsigned(7 DOWNTO 0);

    --------------------------------------------------------------------
    -- Outputs (registered)
    --------------------------------------------------------------------
    SIGNAL img_in_en_r,        img_in_en_next        : STD_LOGIC;
    SIGNAL img_in_addr_r,      img_in_addr_next      : STD_LOGIC_VECTOR(constants.CLAHE_IMG_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL mapping_in_en_r,    mapping_in_en_next    : STD_LOGIC;
    SIGNAL mapping_in_addr_r,  mapping_in_addr_next  : STD_LOGIC_VECTOR(constants.CLAHE_MAPPING_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL clahe_out_en_r,     clahe_out_en_next     : STD_LOGIC;
    SIGNAL clahe_out_addr_r,   clahe_out_addr_next   : STD_LOGIC_VECTOR(constants.CLAHE_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
    SIGNAL clahe_out_d_r,      clahe_out_d_next      : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL rdy_r,              rdy_next              : STD_LOGIC;

BEGIN

    img_in_en       <= img_in_en_r;
    img_in_addr     <= img_in_addr_r;
    mapping_in_en   <= mapping_in_en_r;
    mapping_in_addr <= mapping_in_addr_r;
    clahe_out_en    <= clahe_out_en_r;
    clahe_out_addr  <= clahe_out_addr_r;
    clahe_out_d     <= clahe_out_d_r;
    rdy             <= rdy_r;

    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            state           <= state_next;
            x               <= x_next;
            y               <= y_next;

            pixel_center    <= pixel_center_next;
            tl              <= tl_next;
            tr              <= tr_next;
            bl              <= bl_next;
            br              <= br_next;

            img_in_en_r       <= img_in_en_next;
            img_in_addr_r     <= img_in_addr_next;
            mapping_in_en_r   <= mapping_in_en_next;
            mapping_in_addr_r <= mapping_in_addr_next;
            clahe_out_en_r    <= clahe_out_en_next;
            clahe_out_addr_r  <= clahe_out_addr_next;
            clahe_out_d_r     <= clahe_out_d_next;
            rdy_r             <= rdy_next;
        END IF;
    END PROCESS;


    PROCESS (
        state, trg, x, y,
        pixel_center, tl, tr, bl, br,
        img_in_d, mapping_in_d,
        img_in_en_r, img_in_addr_r,
        mapping_in_en_r, mapping_in_addr_r,
        clahe_out_en_r, clahe_out_addr_r, clahe_out_d_r,
        rdy_r
    )
        VARIABLE cur_row   : integer;
        VARIABLE cur_col   : integer;
        VARIABLE pix_addr  : integer;
        VARIABLE tile_idx  : integer;
        VARIABLE dx        : integer;
        VARIABLE dy        : integer;
        VARIABLE dx_n      : integer;
        VARIABLE dy_n      : integer;
        VARIABLE interp    : integer;
        VARIABLE out_addr  : integer;
        VARIABLE map_addr  : integer;
    BEGIN
        state_next          <= state;
        x_next              <= x;
        y_next              <= y;

        pixel_center_next   <= pixel_center;
        tl_next             <= tl;
        tr_next             <= tr;
        bl_next             <= bl;
        br_next             <= br;

        img_in_en_next       <= '0';
        img_in_addr_next     <= img_in_addr_r;
        mapping_in_en_next   <= '0';
        mapping_in_addr_next <= mapping_in_addr_r;
        clahe_out_en_next    <= '0';
        clahe_out_addr_next  <= clahe_out_addr_r;
        clahe_out_d_next     <= clahe_out_d_r;
        rdy_next             <= '0';

        CASE state IS

            ----------------------------------------------------------------
            WHEN IDLE =>
                IF trg = '1' THEN
                    x_next      <= 0;
                    y_next      <= 0;
                    state_next  <= PIX_REQ;
                END IF;

            ----------------------------------------------------------------
            -- Request center pixel: img[y + PATCH_Y/2, x + PATCH_X/2]
            WHEN PIX_REQ =>
                pix_addr := (y + PATCH_Y/2) * INPUT_X + (x + PATCH_X/2);
                img_in_en_next   <= '1';
                img_in_addr_next <= std_logic_vector(
                    to_unsigned(pix_addr, img_in_addr_next'length)
                );
                state_next       <= PIX_WAIT;

            ----------------------------------------------------------------
            -- Wait one cycle: center pixel available on img_in_d
            WHEN PIX_WAIT =>
                pixel_center_next <= unsigned(img_in_d);
                state_next        <= TL_REQ;

            ----------------------------------------------------------------
            -- Request TL mapping: mapping[cur_row, cur_col][pixel_center]
            WHEN TL_REQ =>
                cur_row  := y / PATCH_Y;
                cur_col  := x / PATCH_X;
                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx * TILE_LUT_SIZE + to_integer(pixel_center);

                mapping_in_en_next   <= '1';
                mapping_in_addr_next <= std_logic_vector(
                    to_unsigned(map_addr, mapping_in_addr_next'length)
                );
                state_next           <= TL_WAIT;

            WHEN TL_WAIT =>
                tl_next   <= unsigned(mapping_in_d);
                state_next <= TR_REQ;

            ----------------------------------------------------------------
            -- TR: mapping[cur_row, cur_col+1][pixel_center]
            WHEN TR_REQ =>
                cur_row  := y / PATCH_Y;
                cur_col  := x / PATCH_X + 1;
                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx * TILE_LUT_SIZE + to_integer(pixel_center);

                mapping_in_en_next   <= '1';
                mapping_in_addr_next <= std_logic_vector(
                    to_unsigned(map_addr, mapping_in_addr_next'length)
                );
                state_next           <= TR_WAIT;

            WHEN TR_WAIT =>
                tr_next   <= unsigned(mapping_in_d);
                state_next <= BL_REQ;

            ----------------------------------------------------------------
            -- BL: mapping[cur_row+1, cur_col][pixel_center]
            WHEN BL_REQ =>
                cur_row  := y / PATCH_Y + 1;
                cur_col  := x / PATCH_X;
                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx * TILE_LUT_SIZE + to_integer(pixel_center);

                mapping_in_en_next   <= '1';
                mapping_in_addr_next <= std_logic_vector(
                    to_unsigned(map_addr, mapping_in_addr_next'length)
                );
                state_next           <= BL_WAIT;

            WHEN BL_WAIT =>
                bl_next   <= unsigned(mapping_in_d);
                state_next <= BR_REQ;

            ----------------------------------------------------------------
            -- BR: mapping[cur_row+1, cur_col+1][pixel_center]
            WHEN BR_REQ =>
                cur_row  := y / PATCH_Y + 1;
                cur_col  := x / PATCH_X + 1;
                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx * TILE_LUT_SIZE + to_integer(pixel_center);

                mapping_in_en_next   <= '1';
                mapping_in_addr_next <= std_logic_vector(
                    to_unsigned(map_addr, mapping_in_addr_next'length)
                );
                state_next           <= BR_WAIT;

            WHEN BR_WAIT =>
                br_next    <= unsigned(mapping_in_d);
                state_next <= COMPUTE;

            ----------------------------------------------------------------
            -- Compute bilinear interpolation (same as Python)
            WHEN COMPUTE =>
                dx   := x MOD PATCH_X;
                dy   := y MOD PATCH_Y;
                dx_n := PATCH_X - 1 - dx;
                dy_n := PATCH_Y - 1 - dy;

                interp :=
                    (to_integer(tl) * dx_n * dy_n +
                     to_integer(tr) * dx   * dy_n +
                     to_integer(bl) * dx_n * dy   +
                     to_integer(br) * dx   * dy)
                    / PATCH_X / PATCH_Y;

                -- clamp to 0..255
                IF interp < 0 THEN
                    interp := 0;
                ELSIF interp > 255 THEN
                    interp := 255;
                END IF;

                out_addr := y * OUT_X + x;

                clahe_out_en_next   <= '1';
                clahe_out_addr_next <= std_logic_vector(
                    to_unsigned(out_addr, clahe_out_addr_next'length)
                );
                clahe_out_d_next    <= std_logic_vector(
                    to_unsigned(interp, 8)
                );

                state_next <= WRITE_OUT;

            ----------------------------------------------------------------
            -- Advance to next output pixel
            WHEN WRITE_OUT =>
                IF x = OUT_X - 1 THEN
                    x_next <= 0;
                    IF y = OUT_Y - 1 THEN
                        state_next <= DONE;
                    ELSE
                        y_next     <= y + 1;
                        state_next <= PIX_REQ;
                    END IF;
                ELSE
                    x_next     <= x + 1;
                    state_next <= PIX_REQ;
                END IF;

            ----------------------------------------------------------------
            WHEN DONE =>
                rdy_next <= '1';
                IF trg = '0' THEN
                    state_next <= IDLE;
                END IF;

        END CASE;
    END PROCESS;

END rtl;
