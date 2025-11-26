LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;

ENTITY top IS
    PORT (
        -- 12 MHz system clock input
        sysclk : IN STD_LOGIC;

        -- camera power
        cam_rst : OUT STD_LOGIC;
        cam_pwdn : OUT STD_LOGIC;
        cam_xclk : OUT STD_LOGIC;

        -- camera config
        cam_scl : OUT STD_LOGIC;
        cam_sda : INOUT STD_LOGIC;

        -- camera data
        cam_pclk : IN STD_LOGIC;
        cam_vsync : IN STD_LOGIC;
        cam_hsync : IN STD_LOGIC;
        cam_d : IN STD_LOGIC_VECTOR(7 DOWNTO 0);

        -- vga output
        vga_vsync : OUT STD_LOGIC;
        vga_hsync : OUT STD_LOGIC;
        vga_r : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
        vga_g : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
        vga_b : OUT STD_LOGIC_VECTOR(1 DOWNTO 0)
    );
END top;

ARCHITECTURE arch OF top IS
    COMPONENT cmt
        PORT (
            sysclk : IN STD_LOGIC;
            cam_xclk : OUT STD_LOGIC
        );
    END COMPONENT;
BEGIN
    -- clock management
    cmt_inst : cmt
    PORT MAP(
        sysclk => sysclk,
        cam_xclk => cam_xclk
    );
END ARCHITECTURE;