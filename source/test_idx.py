import numpy as np

loc_m_n_dims_to_encode = 3

loc_m_n_to_pad = 5

batch_size = 5
fan_out = loc_m_n_dims_to_encode + loc_m_n_to_pad
loc_m_stride = 8

output_acc = [0] * (fan_out * batch_size)
for encoded_index in range(40):
    i = encoded_index // fan_out

    j = encoded_index - i * fan_out
    idx = i * loc_m_stride + j
    if j >= loc_m_n_dims_to_encode:
        output_acc[idx] = 1
    else:
        output_acc[idx] = 2
    print(f"i: {i}, j: {j}, idx: {idx}, encoded_idx: {encoded_index}")

print(output_acc)
