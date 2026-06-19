from pathlib import Path
import numpy as np
from scipy.signal import convolve2d
import cocotb
import cv2

from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge
from cocotb.utils import get_sim_time
from cocotb.triggers import Timer
from matplotlib import pyplot as plt

PIXEL_WIDTH_OUT = 8
FRAC_BITS = 6
KERNEL_NUM = 24

def get_neighbors(input_array, index, width):
    neighbors = []
    x = index % width
    y = index // width

    for i in range(max(0, x - 1), min(width, x + 2)):
        for j in range(max(0, y - 1), min(len(input_array) // width, y + 2)):
            neighbor_index = j * width + i
            neighbors.append(input_array[neighbor_index])
    return neighbors


def get_neighbor_array(image, input_array):
    height, width, _ = image.shape

    array_neighbors = []

    neighbor_count = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            i = y * width + x
            neighbors = get_neighbors(input_array, i, width)
            array_neighbors.append(neighbors)
            neighbor_count += 1
    return array_neighbors


def to_fixed_point(value):
    SCALE = 64  
    return int(value * SCALE)


def create_matrix_3x3(p0, p1, p2, p3, p4, p5, p6, p7, p8, pixel_width):
    p0 = to_fixed_point(p0)
    p1 = to_fixed_point(p1)
    p2 = to_fixed_point(p2)
    p3 = to_fixed_point(p3)
    p4 = to_fixed_point(p4)
    p5 = to_fixed_point(p5)
    p6 = to_fixed_point(p6)
    p7 = to_fixed_point(p7)
    p8 = to_fixed_point(p8)

    def pack_vector_3(a, b, c, width):
        return (a & ((1 << width) - 1)) << (2 * width) | (b & ((1 << width) - 1)) << width | (c & ((1 << width) - 1))

    vector0 = pack_vector_3(p0, p1, p2, pixel_width)
    vector1 = pack_vector_3(p3, p4, p5, pixel_width)
    vector2 = pack_vector_3(p6, p7, p8, pixel_width)

    matrix_3x3_val = (vector0 << (2 * 3 * pixel_width)) | (vector1 << (3 * pixel_width)) | vector2
    return matrix_3x3_val

def decode_q4_6_from_binaryvalue(binary_value):

    raw_value = binary_value.integer
    num_bits = len(binary_value.binstr)  
    is_negative = (raw_value & (1 << (num_bits - 1))) != 0  

    if is_negative:
        raw_value -= (1 << num_bits)  

    integer_part = raw_value >> 6  
    fractional_part = raw_value & 0x3F  
    fractional_value = fractional_part / 2**6  

    return integer_part + fractional_value
    

def get_values(full_data):
    BITS_PER_VALUE = 11
    NUM_VALUES = KERNEL_NUM

    values = []
    for i in range(NUM_VALUES):
        start_bit = i * BITS_PER_VALUE
        end_bit = start_bit + BITS_PER_VALUE
    
        value = full_data[start_bit:end_bit-1]

        # print(f"Index {i} Bin value: {value}")

        value_q4_6 = decode_q4_6_from_binaryvalue(value)
        values.append(value_q4_6)

    return values


def convol2d_sw(matrix_px, kernel):
    convol = matrix_px[0]*kernel[0] + matrix_px[1]*kernel[1] +  matrix_px[2]*kernel[2]+\
                matrix_px[3]*kernel[3] + matrix_px[4]*kernel[4] +  matrix_px[5]*kernel[5]+\
                matrix_px[6]*kernel[6] + matrix_px[7]*kernel[7] +  matrix_px[8]*kernel[8]

    return to_fixed_point(convol)/ (1 << FRAC_BITS)



#-------------------------------Convert RGB image to grayscale------------------------------------------
img_original = cv2.imread('monarch_RGB.jpg', cv2.IMREAD_COLOR) 
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

gray_opencv = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY) 
input_image = cv2.normalize(gray_opencv, None, 0, 255, cv2.NORM_MINMAX) / 255.0

array_input_image = []

for i in range(input_image.shape[0]): 
    for j in range(input_image.shape[1]):
        pixel = input_image[i][j]
        fixed_point_pixel = pixel * (1 << FRAC_BITS) #?
        array_input_image.append(pixel)


# with open('monarch_320x240.txt', 'w') as f:
#     for pixel in array_input_image:
#         f.write(f"{int(str(pixel), 2)}\n")


#----------------------------------------cocotb test bench----------------------------------------------
# Reset
async def reset_dut(dut, duration_ns):
    dut.nreset_i.value = 0
    await Timer(duration_ns, units="ns")
    dut.nreset_i.value = 1
    dut.nreset_i._log.info("Reset complete")

# Wait until output file is completely written
async def wait_file():
    Path('output_image.txt').exists()

async def monitor_px_rdy(dut, px_rdy_o, array, px_out, convol_sw):
    while True:
        await RisingEdge(px_rdy_o)
        await FallingEdge(px_rdy_o)
        await Timer(1, units='ns') 
        values = get_values(px_out.value)
        print(f'\n Output Conv layer 1: {values}')
        error, avg_error = error_cal(convol_sw, values)
        print(f'\n Error:  {error}, Avg error {avg_error:.2f}%')


def error_cal(sw, hw):
    error_abs = np.abs(np.array(sw) - np.array(hw))
    error_rel = np.divide(error_abs, np.abs(sw), out=np.zeros_like(error_abs), where=sw!=0)
    error = error_rel * 100
    
    avg_error = np.mean(error)
    
    return error, avg_error


@cocotb.test()
async def cnn_TB(dut):

    array_neighbors = get_neighbor_array(img_original, array_input_image)
    first_neighbors = array_neighbors[0]

    input_data_val = create_matrix_3x3(*first_neighbors, PIXEL_WIDTH_OUT)
  
    array_kernels = []
    convol_sw = []
    for i in range(KERNEL_NUM):
        random_kernel_val = np.random.rand(9)
        print(f'Kernel {i}) {random_kernel_val}')
        convol_sw.append(convol2d_sw(first_neighbors, random_kernel_val))
        kernel = create_matrix_3x3(*random_kernel_val, PIXEL_WIDTH_OUT)
        array_kernels.append(kernel)

    print(f'\n Convol sw: {convol_sw}')

    # Clock cycle
    clock = Clock(dut.clk_i, 20, units="ns") 
    cocotb.start_soon(clock.start(start_high=False))

    # Inital
    dut.in_value_i.value = 0
    dut.start_cnn_i.value = 0
    dut.px_rdy_i.value = 0
    dut.kernel_valid_i.value = 0

    array_output_image = []

    dut.kernel_in.value = 0

    px_rdy_o = dut.px_rdy_o
    px_out_o = dut.out_px_array

    print(f'\n Input pixels 3x3: {first_neighbors}\n')

    # Start the process to monitor the px_rdy_o signal in parallel
    cocotb.start_soon(monitor_px_rdy(dut, px_rdy_o, array_output_image, px_out_o, convol_sw))

    await reset_dut(dut, 10) 
    await FallingEdge(dut.clk_i)


    for i in range(KERNEL_NUM):
        dut.kernel_valid_i.value = 1
        dut.kernel_in.value = array_kernels[i]
        await FallingEdge(dut.clk_i)
        dut.kernel_valid_i.value = 0
        if i == KERNEL_NUM-1:
            dut.start_cnn_i.value = 1

    for ind, pixel in enumerate(first_neighbors):
        dut.in_value_i.value = to_fixed_point(pixel)
        await FallingEdge(dut.clk_i)
        dut.px_rdy_i.value = 1
        await FallingEdge(dut.clk_i)
        dut.px_rdy_i.value = 0
        if ind%8 == 0:
            print(f'Processed pixels: {ind}')

    # for i, neighbor_array in enumerate(array_neighbors[1:]):
    #     for ind, pixel in enumerate(neighbor_array[6:]):
    #         dut.in_value_i.value = to_fixed_point(pixel)
    #         await FallingEdge(dut.clk_i)
    #         dut.px_rdy_i.value = 1
    #         await FallingEdge(dut.clk_i)
    #         dut.px_rdy_i.value = 0
    #     if i%10000 == 0:
    #          dut._log.info(f'Processed pixels: {i}')

    await FallingEdge(dut.clk_i)
    dut.px_rdy_i.value = 1
    await FallingEdge(dut.clk_i)
    dut.start_cnn_i.value = 0
    await RisingEdge(dut.clk_i)


    

    



    