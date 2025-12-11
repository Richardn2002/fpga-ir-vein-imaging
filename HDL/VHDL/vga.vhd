LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
USE IEEE.numeric_std.ALL;

USE work.constants;

ENTITY vga IS
    -- cannot be paused, upper level should swap input when ram_reading is low
    GENERIC (
        -- reads 90x90 from ram and renders into 450*450 center of 640*480
        CONSTANT INPUT_X : NATURAL := 90;
        CONSTANT INPUT_Y : NATURAL := 90;
        CONSTANT PIXSIZE : NATURAL := 5;
        CONSTANT HBEGIN : NATURAL := 95;
        CONSTANT HEND : NATURAL := 545;
        CONSTANT VBEGIN : NATURAL := 15;
        CONSTANT VEND : NATURAL := 465
    );
    PORT (
        clk : IN STD_LOGIC;

        r : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
        g : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
        b : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
        hsync : OUT STD_LOGIC;
        vsync : OUT STD_LOGIC;

        ram_re : OUT STD_LOGIC;
        ram_addr : OUT STD_LOGIC_VECTOR(constants.HESSIAN_OUTPUT_ADDR_BITS - 1 DOWNTO 0);
        ram_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        ram_reading : OUT STD_LOGIC -- high on first pixel read until last pixel read
    );
END vga;

ARCHITECTURE arch OF vga IS
    SIGNAL hcount : NATURAL RANGE 0 TO 799;
    SIGNAL vcount : NATURAL RANGE 0 TO 524;
    SIGNAL blank : STD_LOGIC;

    SIGNAL pixel : STD_LOGIC_VECTOR(1 DOWNTO 0);
    SIGNAL x : NATURAL RANGE 0 TO INPUT_X - 1;
    SIGNAL y : NATURAL RANGE 0 TO INPUT_Y - 1;

    SIGNAL ram_reading_next : STD_LOGIC;
BEGIN
    ------------------------------------------------------------------
    -- VGA display counters
    --
    -- Pixel clock: 25.175 MHz (actual: 25.17550 MHz)
    -- Horizontal count (active low sync):
    --     0 to 639: Active video
    --     640 to 799: Horizontal blank
    --     656 to 751: Horizontal sync (active low)
    -- Vertical count (active low sync):
    --     0 to 479: Active video
    --     480 to 524: Vertical blank
    --     490 to 491: Vertical sync (active low)
    ------------------------------------------------------------------
    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            -- Pixel position counters
            IF (hcount >= 799) THEN
                hcount <= 0;
                IF (vcount >= 524) THEN
                    vcount <= 0;
                ELSE
                    vcount <= vcount + 1;
                END IF;
            ELSE
                hcount <= hcount + 1;
            END IF;
            -- Sync, blank and frame
            IF (hcount >= 656) AND
                (hcount <= 751) THEN
                hsync <= '0';
            ELSE
                hsync <= '1';
            END IF;
            IF (vcount >= 490) AND
                (vcount <= 491) THEN
                vsync <= '0';
            ELSE
                vsync <= '1';
            END IF;
            IF (hcount >= 640) OR
                (vcount >= 480) THEN
                blank <= '1';
            ELSE
                blank <= '0';
            END IF;
        END IF;
    END PROCESS;
    ------------------------------------------------------------------
    -- VGA output with blanking
    ------------------------------------------------------------------
    r <= b"00" WHEN blank = '1' ELSE
        pixel;
    g <= b"00" WHEN blank = '1' ELSE
        pixel;
    b <= b"00" WHEN blank = '1' ELSE
        pixel;

    -- convert to 2-bit color, valid region only
    pixel <= ram_d(7 DOWNTO 6) WHEN hcount >= HBEGIN AND hcount < HEND AND vcount >= VBEGIN AND vcount < VEND
        ELSE
        "00";
    -- read pixel data, 1 cycle early for ram delay
    x <= (hcount - HBEGIN + 1) / PIXSIZE;
    y <= (vcount - VBEGIN) / PIXSIZE;
    ram_addr <= STD_LOGIC_VECTOR(to_unsigned(y * INPUT_X + x, constants.HESSIAN_OUTPUT_ADDR_BITS));
    PROCESS (hcount, vcount)
    BEGIN
        -- let registered ram_reading be high when address of first pixel is being output
        -- until (include) address of last pixel is being output
        IF (vcount = VBEGIN AND hcount >= HBEGIN - 2) OR (vcount > VBEGIN AND vcount < VEND - 1) OR (vcount = VEND - 1 AND hcount <= hend - 3) THEN
            ram_reading_next <= '1';
        ELSE
            ram_reading_next <= '0';
        END IF;

        -- drive ram read enable during valid pixel region
        IF (HBEGIN <= hcount + 1 AND hcount + 1 < HEND AND VBEGIN <= vcount AND vcount < VEND) THEN
            ram_re <= '1';
        ELSE
            ram_re <= '0';
        END IF;
    END PROCESS;

    PROCESS (clk) BEGIN
        IF rising_edge(clk) THEN
            ram_reading <= ram_reading_next;
        END IF;
    END PROCESS;
END arch;