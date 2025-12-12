LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

--USE work.constants;

ENTITY CLAHE_output IS
    PORT (
        clk : IN STD_LOGIC;
        trg : IN STD_LOGIC;
        rdy : OUT STD_LOGIC;

        -- Input image RAM interface
        img_in_en : OUT STD_LOGIC;
        img_in_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        img_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        -- CLAHE mapping LUT RAM interface
        mapping_in_en : OUT STD_LOGIC;
        -- mapping_in_addr : out std_logic_vector(constants.CLAHE_MAPPING_ADDR_BITS - 1 downto 0);
        mapping_in_addr : OUT STD_LOGIC_VECTOR(11 DOWNTO 0);
        mapping_in_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        -- CLAHE output RAM interface
        clahe_out_en : OUT STD_LOGIC;
        clahe_out_addr : OUT STD_LOGIC_VECTOR(13 DOWNTO 0);
        clahe_out_d : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END CLAHE_output;

ARCHITECTURE rtl OF CLAHE_output IS

    --------------------------------------------------------------------
    -- Constants
    --------------------------------------------------------------------

    CONSTANT INPUT_X : INTEGER := 128;
    CONSTANT INPUT_Y : INTEGER := 128;

    CONSTANT PATCH_X : INTEGER := 32;
    CONSTANT PATCH_Y : INTEGER := 32;
    CONSTANT NUM_X : INTEGER := 4;
    CONSTANT NUM_Y : INTEGER := 4;

    CONSTANT OUT_X : INTEGER := 96; -- 96
    CONSTANT OUT_Y : INTEGER := 96;

    CONSTANT TILE_LUT_SIZE : INTEGER := 256;

    --------------------------------------------------------------------
    -- State machine
    --------------------------------------------------------------------

    TYPE state_t IS (
        IDLE,
        PIX_REQ, PIX_WAIT,
        TL_REQ, TL_WAIT,
        TR_REQ, TR_WAIT,
        BL_REQ, BL_WAIT,
        BR_REQ, BR_WAIT,
        COMPUTE,
        WRITE_OUT,
        DONE
    );

    SIGNAL state, state_next : state_t;

    --------------------------------------------------------------------
    -- Pixel coordinates
    --------------------------------------------------------------------

    SIGNAL x, x_next : INTEGER RANGE 0 TO OUT_X - 1;
    SIGNAL y, y_next : INTEGER RANGE 0 TO OUT_Y - 1;

    --------------------------------------------------------------------
    -- Pixel + LUT values (registered)
    --------------------------------------------------------------------

    SIGNAL pixel_center, pixel_center_next : unsigned(7 DOWNTO 0);
    SIGNAL tl, tl_next : unsigned(7 DOWNTO 0);
    SIGNAL tr, tr_next : unsigned(7 DOWNTO 0);
    SIGNAL bl, bl_next : unsigned(7 DOWNTO 0);
    SIGNAL br, br_next : unsigned(7 DOWNTO 0);

BEGIN

    --------------------------------------------------------------------
    -- Sequential logic (state + registers only)
    --------------------------------------------------------------------
    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            state <= state_next;
            x <= x_next;
            y <= y_next;

            pixel_center <= pixel_center_next;
            tl <= tl_next;
            tr <= tr_next;
            bl <= bl_next;
            br <= br_next;
        END IF;
    END PROCESS;

    PROCESS (
        state, trg, x, y,
        pixel_center, tl, tr, bl, br,
        img_in_d, mapping_in_d
        )
        VARIABLE cur_row : INTEGER;
        VARIABLE cur_col : INTEGER;
        VARIABLE pix_addr : INTEGER;
        VARIABLE tile_idx : INTEGER;
        VARIABLE map_addr : INTEGER;
        VARIABLE dx, dy, dx_n, dy_n : INTEGER;
        VARIABLE interp : INTEGER;
        VARIABLE out_addr : INTEGER;
    BEGIN

        ----------------------------------------------------------------
        -- Default signal assignments
        ----------------------------------------------------------------
        state_next <= state;

        x_next <= x;
        y_next <= y;

        pixel_center_next <= pixel_center;
        tl_next <= tl;
        tr_next <= tr;
        bl_next <= bl;
        br_next <= br;

        img_in_en <= '0';
        img_in_addr <= (OTHERS => '0');

        mapping_in_en <= '0';
        mapping_in_addr <= (OTHERS => '0');

        clahe_out_en <= '0';
        clahe_out_addr <= (OTHERS => '0');
        clahe_out_d <= (OTHERS => '0');

        rdy <= '0';
        ----------------------------------------------------------------
        -- State Machine
        ----------------------------------------------------------------
        CASE state IS
                ------------------------------------------------------------
            WHEN IDLE =>
                IF trg = '1' THEN
                    x_next <= 0;
                    y_next <= 0;
                    state_next <= PIX_REQ;
                END IF;

                ------------------------------------------------------------
                -- Request the center pixel
                ------------------------------------------------------------
            WHEN PIX_REQ =>
                pix_addr :=
                    (y + PATCH_Y/2) * INPUT_X +
                    (x + PATCH_X/2);

                img_in_en <= '1';
                img_in_addr <= STD_LOGIC_VECTOR(
                    to_unsigned(pix_addr, img_in_addr'length));

                state_next <= PIX_WAIT;
            WHEN PIX_WAIT =>
                -- One cycle later: pixel is ready
                pixel_center_next <= unsigned(img_in_d);
                state_next <= TL_REQ;

                ------------------------------------------------------------
                -- TL LUT
                ------------------------------------------------------------
            WHEN TL_REQ =>
                cur_row := y / PATCH_Y;
                cur_col := x / PATCH_X;

                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx * TILE_LUT_SIZE +
                    to_integer(pixel_center);

                mapping_in_en <= '1';
                mapping_in_addr <= STD_LOGIC_VECTOR(
                    to_unsigned(map_addr, mapping_in_addr'length));

                state_next <= TL_WAIT;

            WHEN TL_WAIT =>
                tl_next <= unsigned(mapping_in_d);
                state_next <= TR_REQ;

                ------------------------------------------------------------
                -- TR
                ------------------------------------------------------------
            WHEN TR_REQ =>
                cur_row := y / PATCH_Y;
                cur_col := x / PATCH_X + 1;

                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx * TILE_LUT_SIZE +
                    to_integer(pixel_center);

                mapping_in_en <= '1';
                mapping_in_addr <= STD_LOGIC_VECTOR(
                    to_unsigned(map_addr, mapping_in_addr'length));

                state_next <= TR_WAIT;

            WHEN TR_WAIT =>
                tr_next <= unsigned(mapping_in_d);
                state_next <= BL_REQ;

                ------------------------------------------------------------
                -- BL
                ------------------------------------------------------------
            WHEN BL_REQ =>
                cur_row := y / PATCH_Y + 1;
                cur_col := x / PATCH_X;

                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx * TILE_LUT_SIZE +
                    to_integer(pixel_center);

                mapping_in_en <= '1';
                mapping_in_addr <= STD_LOGIC_VECTOR(
                    to_unsigned(map_addr, mapping_in_addr'length));

                state_next <= BL_WAIT;

            WHEN BL_WAIT =>
                bl_next <= unsigned(mapping_in_d);
                state_next <= BR_REQ;

                ------------------------------------------------------------
                -- BR
                ------------------------------------------------------------
            WHEN BR_REQ =>
                cur_row := y / PATCH_Y + 1;
                cur_col := x / PATCH_X + 1;

                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx * TILE_LUT_SIZE +
                    to_integer(pixel_center);

                mapping_in_en <= '1';
                mapping_in_addr <= STD_LOGIC_VECTOR(
                    to_unsigned(map_addr, mapping_in_addr'length));

                state_next <= BR_WAIT;

            WHEN BR_WAIT =>
                br_next <= unsigned(mapping_in_d);
                state_next <= COMPUTE;

                ------------------------------------------------------------
                -- Bilinear interpolation
                ------------------------------------------------------------
            WHEN COMPUTE =>

                dx := x MOD PATCH_X;
                dy := y MOD PATCH_Y;
                dx_n := PATCH_X - 1 - dx;
                dy_n := PATCH_Y - 1 - dy;

                interp :=
                    (to_integer(tl) * dx_n * dy_n +
                    to_integer(tr) * dx * dy_n +
                    to_integer(bl) * dx_n * dy +
                    to_integer(br) * dx * dy)
                    / PATCH_X / PATCH_Y;

                IF interp < 0 THEN
                    interp := 0;
                ELSIF interp > 255 THEN
                    interp := 255;
                END IF;

                out_addr := y * OUT_X + x;

                clahe_out_en <= '1';
                clahe_out_addr <= STD_LOGIC_VECTOR(
                    to_unsigned(out_addr, clahe_out_addr'length));
                clahe_out_d <= STD_LOGIC_VECTOR(
                    to_unsigned(interp, 8));

                state_next <= WRITE_OUT;
                ------------------------------------------------------------
                -- Move to next pixel
                ------------------------------------------------------------
            WHEN WRITE_OUT =>
                IF x = OUT_X - 1 THEN
                    x_next <= 0;

                    IF y = OUT_Y - 1 THEN
                        state_next <= DONE;
                    ELSE
                        y_next <= y + 1;
                        state_next <= PIX_REQ;
                    END IF;

                ELSE
                    x_next <= x + 1;
                    state_next <= PIX_REQ;
                END IF;

                ------------------------------------------------------------
            WHEN DONE =>
                rdy <= '1';
                IF trg = '0' THEN
                    state_next <= IDLE;
                END IF;
        END CASE;
    END PROCESS;

END rtl;