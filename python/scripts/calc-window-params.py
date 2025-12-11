import sys

if len(sys.argv) < 5:
    print("Usage: python calc-window-params.py <hbegin> <hend> <vbegin> <vend>")
    sys.exit(-1)

hbegin = int(sys.argv[1])
hend = int(sys.argv[2])
vbegin = int(sys.argv[3])
vend = int(sys.argv[4])

h_edge_offset = 0b10
hstart = (168 + hbegin * 2) % 784
hstop = (168 + hend * 2) % 784
vstart = 12 + vbegin * 2
vstop = 12 + vend * 2

print('x"%02x%02x"' % (0x17, hstart >> 3))
print('x"%02x%02x"' % (0x18, hstop >> 3))
print(
    'x"%02x%02x"'
    % (
        0x32,
        ((h_edge_offset << 6) & 0b11000000)
        + ((hstop << 3) & 0b00111000)
        + (hstart & 0b00000111),
    )
)
print('x"%02x%02x"' % (0x19, vstart >> 2))
print('x"%02x%02x"' % (0x1A, vstop >> 2))
print('x"%02x%02x"' % (0x03, ((vstop << 2) & 0b00001100) + (vstart & 0b00000011)))
