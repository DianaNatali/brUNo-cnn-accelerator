"""
cnn_TB.py  —  Testbench cocotb para top_conv1_pool1
Pipeline: Conv1 (24 kernels 3×3 + ReLU) → MaxPool1 (2×2) → LineBuffer

Flujo:
  1. Cargar 24 kernels aleatorios
  2. Enviar imagen completa píxel a píxel (px_rdy_i + in_px_i)
  3. Capturar salidas de Conv1 y Pool1
  4. Guardar feature maps como imágenes PNG
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------
# Parámetros (deben coincidir con parameters.svh)
# ---------------------------------------------------------------------------
PIXEL_WIDTH_OUT = 8
FRAC_BITS       = 6
BITS_Q4_6       = 11
KERNEL_NUM      = 24
SCALE           = 1 << FRAC_BITS   # 64


# ---------------------------------------------------------------------------
# Helpers de punto fijo
# ---------------------------------------------------------------------------
def to_fixed_point(value: float) -> int:
    """Float [0,1] → Q1.6 entero (8 bits)"""
    return int(np.clip(value * SCALE, 0, 255))


def fp_to_float(raw: int, num_bits: int = BITS_Q4_6) -> float:
    """Entero en complemento a 2 (Q4.6, 11 bits) → float"""
    if raw & (1 << (num_bits - 1)):
        raw -= 1 << num_bits
    return raw / SCALE


def decode_conv1_out(bv) -> list:
    """
    Decodifica out_px_array (KERNEL_NUM × BITS_Q4_6 bits) → lista de floats.
    """
    raw = int(bv.value)
    values = []
    for i in range(KERNEL_NUM):
        chunk = (raw >> (i * BITS_Q4_6)) & ((1 << BITS_Q4_6) - 1)
        values.append(fp_to_float(chunk))
    return values


# ---------------------------------------------------------------------------
# Packing de kernel (matrix_3x3_8bits packed struct)
# ---------------------------------------------------------------------------
def pack_kernel(k: np.ndarray) -> int:
    """
    k: array de 9 floats [0,1]
    Retorna entero que representa matrix_3x3_8bits packed:
      { vector2, vector1, vector0 }  cada vector = { p2, p1, p0 }
    """
    def pack_vec(a, b, c):
        a = to_fixed_point(a) & 0xFF
        b = to_fixed_point(b) & 0xFF
        c = to_fixed_point(c) & 0xFF
        return (a << 16) | (b << 8) | c

    v0 = pack_vec(k[0], k[1], k[2])
    v1 = pack_vec(k[3], k[4], k[5])
    v2 = pack_vec(k[6], k[7], k[8])
    return (v0 << 48) | (v1 << 24) | v2


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
async def reset_dut(dut, duration_ns: int = 20):
    dut.nreset_i.value = 0
    await Timer(duration_ns, units="ns")
    dut.nreset_i.value = 1
    dut._log.info("Reset completo")


# ---------------------------------------------------------------------------
# Monitor Conv1
# ---------------------------------------------------------------------------
async def monitor_conv1(dut, feature_maps: list, total_px: int, W_conv: int):
    """
    Captura píxeles de Conv1 en orden raster.
    feature_maps: lista de KERNEL_NUM arrays 2D ya inicializados.
    """
    for px_idx in range(total_px):
        await RisingEdge(dut.conv1_rdy_o)
        await Timer(1, units='ns')
        values = decode_conv1_out(dut.conv1_out_o)
        y = px_idx // W_conv
        x = px_idx  % W_conv
        for ch in range(KERNEL_NUM):
            feature_maps[ch][y, x] = values[ch]

    dut._log.info(f"Conv1: {total_px} píxeles capturados")


# ---------------------------------------------------------------------------
# Monitor Pool1
# ---------------------------------------------------------------------------
async def monitor_pool1(dut, pool_maps: list, total_px: int, W_pool: int):
    """
    Captura píxeles de Pool1 en orden raster.
    pool1_out_o es un bus plano KERNEL_NUM*BITS_Q4_6 bits.
    Canal ch ocupa bits [ch*BITS_Q4_6 +: BITS_Q4_6].
    """
    mask = (1 << BITS_Q4_6) - 1
    for px_idx in range(total_px):
        await RisingEdge(dut.pool1_rdy_o)
        await Timer(1, units='ns')
        bus = int(dut.pool1_out_o.value)
        y = px_idx // W_pool
        x = px_idx  % W_pool
        for ch in range(KERNEL_NUM):
            raw = (bus >> (ch * BITS_Q4_6)) & mask
            pool_maps[ch][y, x] = fp_to_float(raw)

    dut._log.info(f"Pool1: {total_px} píxeles capturados")


# ---------------------------------------------------------------------------
# Guardar feature maps
# ---------------------------------------------------------------------------
def save_feature_maps(maps: list, title: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cols, rows = 6, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(title, fontsize=14)

    for idx, (ax, fmap) in enumerate(zip(axes.flat, maps)):
        ax.imshow(fmap, cmap='viridis', aspect='auto')
        ax.set_title(f'ch {idx}', fontsize=9)
        ax.axis('off')
        norm = cv2.normalize(fmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(out_dir / f'ch_{idx:02d}.png'), norm)

    plt.tight_layout()
    fig.savefig(str(out_dir / 'mosaic.png'), dpi=120)
    plt.close(fig)
    print(f"[TB] Guardado: {out_dir}/mosaic.png")


# ---------------------------------------------------------------------------
# Test principal
# ---------------------------------------------------------------------------
@cocotb.test()
async def TB_cnn_first_layer(dut):

    # ------------------------------------------------------------------
    # 1. Cargar imagen
    # ------------------------------------------------------------------
    img_path = 'monarch_RGB.jpg'
    assert Path(img_path).exists(), f"No se encontró {img_path}"

    img_bgr  = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_norm = cv2.normalize(img_gray, None, 0.0, 1.0,
                             cv2.NORM_MINMAX).astype(np.float64)

    H, W   = img_norm.shape
    H_conv = H - 2
    W_conv = W - 2
    H_pool = H_conv // 2
    W_pool = W_conv // 2

    dut._log.info(f"Imagen: {H}×{W}  Conv1: {H_conv}×{W_conv}  Pool1: {H_pool}×{W_pool}")

    # ------------------------------------------------------------------
    # 2. Kernels aleatorios
    # ------------------------------------------------------------------
    np.random.seed(42)
    kernels = [np.random.rand(3, 3) for _ in range(KERNEL_NUM)]

    # ------------------------------------------------------------------
    # 3. Inicializar DUT
    # ------------------------------------------------------------------
    clock = Clock(dut.clk_i, 20, units="ns")
    cocotb.start_soon(clock.start(start_high=False))

    dut.in_px_i.value        = 0
    dut.start_cnn_i.value    = 0
    dut.px_rdy_i.value       = 0
    dut.kernel_valid_i.value = 0
    dut.kernel_in.value      = 0

    await reset_dut(dut, 20)
    await FallingEdge(dut.clk_i)

    # ------------------------------------------------------------------
    # 4. Cargar kernels
    # ------------------------------------------------------------------
    dut._log.info("Cargando kernels...")
    for i, k in enumerate(kernels):
        dut.kernel_valid_i.value = 1
        dut.kernel_in.value      = pack_kernel(k.flatten())
        await FallingEdge(dut.clk_i)
        dut.kernel_valid_i.value = 0
        await FallingEdge(dut.clk_i)

    # Esperar 2 ciclos extra para que kernels_ready_o se estabilice
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.start_cnn_i.value = 1
    dut._log.info("Kernels listos, pipeline activo")

    # ------------------------------------------------------------------
    # 5. Lanzar monitores en paralelo
    # ------------------------------------------------------------------
    conv1_maps = [np.zeros((H_conv, W_conv)) for _ in range(KERNEL_NUM)]
    pool1_maps = [np.zeros((H_pool, W_pool)) for _ in range(KERNEL_NUM)]

    mon_conv1 = cocotb.start_soon(
        monitor_conv1(dut, conv1_maps, H_conv * W_conv, W_conv))
    mon_pool1 = cocotb.start_soon(
        monitor_pool1(dut, pool1_maps, H_pool * W_pool, W_pool))

    # ------------------------------------------------------------------
    # 6. Enviar imagen píxel a píxel en orden raster
    # ------------------------------------------------------------------
    dut._log.info(f"Enviando imagen ({H*W} píxeles)...")
    for y in range(H):
        for x in range(W):
            dut.in_px_i.value  = to_fixed_point(img_norm[y, x])
            dut.px_rdy_i.value = 1
            await FallingEdge(dut.clk_i)
            dut.px_rdy_i.value = 0
            await FallingEdge(dut.clk_i)

        if y % 20 == 0:
            dut._log.info(f"  Fila {y}/{H}")

    # ------------------------------------------------------------------
    # 7. Esperar a que los monitores terminen
    # ------------------------------------------------------------------
    await mon_conv1
    await mon_pool1
    dut.start_cnn_i.value = 0

    # ------------------------------------------------------------------
    # 8. Guardar feature maps
    # ------------------------------------------------------------------
    out_root = Path("feature_maps")
    save_feature_maps(conv1_maps, "Conv1", out_root / "conv1")
    save_feature_maps(pool1_maps, "Pool1", out_root / "pool1")

    dut._log.info("Simulación completa. Feature maps en feature_maps/")