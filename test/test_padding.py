
def to_packed_layout_coord(idx, rows, cols):
    i = idx // cols
    j = idx % cols
    if (i % 2) == 0:
        return i * cols + 2 * j
    else:
        return (i - 1) * cols + 2 * j + 1

# for i in range(4096):
print(
    to_packed_layout_coord(
        382, 64, 64
    )
)
