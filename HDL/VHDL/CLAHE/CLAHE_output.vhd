library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity CLAHE_output is
    port (
        clk  : in  std_logic;
        trg  : in  std_logic;
        rdy  : out std_logic;

        -- Input image RAM interface
        img_in_en   : out std_logic;
        img_in_addr : out std_logic_vector(13 downto 0);
        img_in_d    : in  std_logic_vector(7 downto 0);

        -- CLAHE mapping LUT RAM interface
        mapping_in_en   : out std_logic;
        mapping_in_addr : out std_logic_vector(11 downto 0);
        mapping_in_d    : in  std_logic_vector(7 downto 0);

        -- CLAHE output RAM interface
        clahe_out_en   : out std_logic;
        clahe_out_addr : out std_logic_vector(13 downto 0);
        clahe_out_d    : out std_logic_vector(7 downto 0)
    );
end CLAHE_output;



architecture rtl of CLAHE_output is


    constant INPUT_X  : integer := 128;      -- 128
    constant INPUT_Y  : integer := 128;

    constant PATCH_X  : integer := 32;   -- 32
    constant PATCH_Y  : integer := 32;
    constant NUM_X    : integer := 4;   -- 4
    constant NUM_Y    : integer := 4;

    constant OUT_X    : integer := 96;     -- 96
    constant OUT_Y    : integer := 96;     

    constant TILE_LUT_SIZE : integer := 256;
    
    type state_t is (
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

    signal state, state_next : state_t;
    signal x, x_next : integer range 0 to OUT_X - 1;
    signal y, y_next : integer range 0 to OUT_Y - 1;

    signal pixel_center, pixel_center_next : unsigned(7 downto 0);
    signal tl, tl_next : unsigned(7 downto 0);
    signal tr, tr_next : unsigned(7 downto 0);
    signal bl, bl_next : unsigned(7 downto 0);
    signal br, br_next : unsigned(7 downto 0);

begin

    process(clk)
    begin
        if rising_edge(clk) then
            state <= state_next;
            x     <= x_next;
            y     <= y_next;

            pixel_center <= pixel_center_next;
            tl <= tl_next;
            tr <= tr_next;
            bl <= bl_next;
            br <= br_next;
        end if;
    end process;

    process(
        state, trg, x, y,
        pixel_center, tl, tr, bl, br,
        img_in_d, mapping_in_d
    )
        variable cur_row  : integer;
        variable cur_col  : integer;
        variable pix_addr : integer;
        variable tile_idx : integer;
        variable map_addr : integer;
        variable dx, dy, dx_n, dy_n : integer;
        variable interp : integer;
        variable out_addr : integer;
    begin

        state_next <= state;

        x_next <= x;
        y_next <= y;

        pixel_center_next <= pixel_center;
        tl_next <= tl;
        tr_next <= tr;
        bl_next <= bl;
        br_next <= br;

        img_in_en   <= '0';
        img_in_addr <= (others => '0');

        mapping_in_en   <= '0';
        mapping_in_addr <= (others => '0');

        clahe_out_en   <= '0';
        clahe_out_addr <= (others => '0');
        clahe_out_d    <= (others => '0');

        rdy <= '0';


        case state is

            when IDLE =>
                if trg = '1' then
                    x_next <= 0;
                    y_next <= 0;
                    state_next <= PIX_REQ;
                end if;

            when PIX_REQ =>
                pix_addr :=
                    (y + PATCH_Y/2) * INPUT_X +
                     (x + PATCH_X/2);

                img_in_en <= '1';
                img_in_addr <= std_logic_vector(
                    to_unsigned(pix_addr, img_in_addr'length));

                state_next <= PIX_WAIT;


            when PIX_WAIT =>
                pixel_center_next <= unsigned(img_in_d);
                state_next <= TL_REQ;

            when TL_REQ =>
                cur_row := y / PATCH_Y;
                cur_col := x / PATCH_X;

                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx*TILE_LUT_SIZE +
                            to_integer(pixel_center);

                mapping_in_en <= '1';
                mapping_in_addr <= std_logic_vector(
                    to_unsigned(map_addr, mapping_in_addr'length));

                state_next <= TL_WAIT;

            when TL_WAIT =>
                tl_next <= unsigned(mapping_in_d);
                state_next <= TR_REQ;

            when TR_REQ =>
                cur_row := y / PATCH_Y;
                cur_col := x / PATCH_X + 1;

                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx*TILE_LUT_SIZE +
                            to_integer(pixel_center);

                mapping_in_en <= '1';
                mapping_in_addr <= std_logic_vector(
                    to_unsigned(map_addr,mapping_in_addr'length));

                state_next <= TR_WAIT;

            when TR_WAIT =>
                tr_next <= unsigned(mapping_in_d);
                state_next <= BL_REQ;

            when BL_REQ =>
                cur_row := y / PATCH_Y + 1;
                cur_col := x / PATCH_X;

                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx*TILE_LUT_SIZE +
                            to_integer(pixel_center);

                mapping_in_en <= '1';
                mapping_in_addr <= std_logic_vector(
                    to_unsigned(map_addr,mapping_in_addr'length));

                state_next <= BL_WAIT;

            when BL_WAIT =>
                bl_next <= unsigned(mapping_in_d);
                state_next <= BR_REQ;

            when BR_REQ =>
                cur_row := y / PATCH_Y + 1;
                cur_col := x / PATCH_X + 1;

                tile_idx := cur_row * NUM_X + cur_col;
                map_addr := tile_idx*TILE_LUT_SIZE +
                            to_integer(pixel_center);

                mapping_in_en <= '1';
                mapping_in_addr <= std_logic_vector(
                    to_unsigned(map_addr,mapping_in_addr'length));

                state_next <= BR_WAIT;

            when BR_WAIT =>
                br_next <= unsigned(mapping_in_d);
                state_next <= COMPUTE;

            when COMPUTE =>

                dx := x mod PATCH_X;
                dy := y mod PATCH_Y;
                dx_n := PATCH_X - 1 - dx;
                dy_n := PATCH_Y - 1 - dy;

                interp :=
                    (to_integer(tl) * dx_n * dy_n +
                     to_integer(tr) * dx   * dy_n +
                     to_integer(bl) * dx_n * dy   +
                     to_integer(br) * dx   * dy)
                    / PATCH_X / PATCH_Y;

                if interp < 0 then
                    interp := 0;
                elsif interp > 255 then
                    interp := 255;
                end if;

                out_addr := y * OUT_X + x;

                clahe_out_en <= '1';
                clahe_out_addr <= std_logic_vector(
                    to_unsigned(out_addr, clahe_out_addr'length));
                clahe_out_d <= std_logic_vector(
                    to_unsigned(interp, 8));

                state_next <= WRITE_OUT;


            when WRITE_OUT =>
                if x = OUT_X - 1 then
                    x_next <= 0;

                    if y = OUT_Y - 1 then
                        state_next <= DONE;
                    else
                        y_next <= y + 1;
                        state_next <= PIX_REQ;
                    end if;

                else
                    x_next <= x + 1;
                    state_next <= PIX_REQ;
                end if;


            when DONE =>
                rdy <= '1';
                if trg = '0' then
                    state_next <= IDLE;
                end if;


        end case;
    end process;

end rtl;
