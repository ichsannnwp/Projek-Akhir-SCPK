import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="SPK Laptop – Fuzzy Tsukamoto", layout="wide")

# ─────────────────────────────────────────────────────────────────────
# KONVERSI TIPE DATA
# ─────────────────────────────────────────────────────────────────────
def konversi_storage(x):
    text = str(x).strip().upper()
    angka_text = "".join([c for c in text if c.isdigit()])
    if not angka_text: return 0.0
    angka = float(angka_text)
    return angka * 1024 if "TB" in text else angka

def konversi_ram(x):
    text = str(x).strip().upper()
    angka_text = "".join([c for c in text if c.isdigit()])
    return float(angka_text) if angka_text else 0.0

# ─────────────────────────────────────────────────────────────────────
# PROFIL PENGGUNA (spek minimum & bobot tiap profil)
# ─────────────────────────────────────────────────────────────────────
PROFIL = {
    "🎮 Gamer": {
        "deskripsi": "Membutuhkan GPU & CPU tinggi untuk gaming lancar di resolusi tinggi.",
        "min_cpu":     20000,   # multiScore
        "min_gpu":     20000,   # 3DMark
        "min_ram":     16,      # GB
        "min_storage": 512,     # GB
        "max_harga":   3000,    # USD
        "bobot": [3, 5, 3, 2, 2],   # CPU, GPU, RAM, Storage, Harga
        "warna": "#e74c3c",
    },
    "🎓 Pelajar / Mahasiswa": {
        "deskripsi": "Butuh laptop terjangkau untuk tugas, browsing, dan produktivitas ringan.",
        "min_cpu":     10000,
        "min_gpu":     0,
        "min_ram":     8,
        "min_storage": 256,
        "max_harga":   800,
        "bobot": [3, 1, 3, 2, 5],
        "warna": "#3498db",
    },
    "🎨 Kreator Konten / Editor": {
        "deskripsi": "Rendering video & foto membutuhkan CPU, RAM, dan storage besar.",
        "min_cpu":     22000,
        "min_gpu":     10000,
        "min_ram":     32,
        "min_storage": 1024,
        "max_harga":   4000,
        "bobot": [5, 3, 5, 4, 2],
        "warna": "#9b59b6",
    },
    "💼 Profesional / Bisnis": {
        "deskripsi": "Multitasking kantor, keamanan, dan portabilitas menjadi prioritas.",
        "min_cpu":     15000,
        "min_gpu":     0,
        "min_ram":     16,
        "min_storage": 512,
        "max_harga":   2000,
        "bobot": [4, 1, 4, 3, 4],
        "warna": "#27ae60",
    },
}

# ─────────────────────────────────────────────────────────────────────
# FUNGSI KEANGGOTAAN FUZZY (Bahu Linear)
# ─────────────────────────────────────────────────────────────────────
def mf_naik(x, a, b):
    """Bahu kanan: 0 di a, 1 di b (TINGGI / MAHAL)"""
    if x <= a: return 0.0
    if x >= b: return 1.0
    return (x - a) / (b - a)

def mf_turun(x, a, b):
    """Bahu kiri: 1 di a, 0 di b (RENDAH / MURAH)"""
    if x <= a: return 1.0
    if x >= b: return 0.0
    return (b - x) / (b - a)

def invers_naik(alpha, a, b):
    return a + np.clip(alpha, 0.0, 1.0) * (b - a)

def invers_turun(alpha, a, b):
    return b - np.clip(alpha, 0.0, 1.0) * (b - a)

# ─────────────────────────────────────────────────────────────────────
# PARAMETER MF (batas berdasarkan konteks domain)
# MF menggunakan batas yang semantik: titik "mulai layak" & "sudah ideal"
# ─────────────────────────────────────────────────────────────────────
def bangun_params(profil_key):
    """Parameter batas MF disesuaikan per profil pengguna."""
    p = PROFIL[profil_key]
    return {
        # CPU: mulai layak di min_cpu, ideal 110% di atas min
        "cpu_rendah_a": p["min_cpu"] * 0.5,
        "cpu_rendah_b": p["min_cpu"] * 1.1,
        "cpu_tinggi_a": p["min_cpu"] * 0.9,
        "cpu_tinggi_b": p["min_cpu"] * 1.5,
        # GPU
        "gpu_rendah_a": max(p["min_gpu"] * 0.5, 1),
        "gpu_rendah_b": max(p["min_gpu"] * 1.1, 100),
        "gpu_tinggi_a": max(p["min_gpu"] * 0.9, 1),
        "gpu_tinggi_b": max(p["min_gpu"] * 1.5, 100),
        # RAM
        "ram_rendah_a": p["min_ram"] * 0.5,
        "ram_rendah_b": p["min_ram"] * 1.25,
        "ram_tinggi_a": p["min_ram"] * 0.75,
        "ram_tinggi_b": p["min_ram"] * 2.0,
        # Storage
        "stor_rendah_a": p["min_storage"] * 0.5,
        "stor_rendah_b": p["min_storage"] * 1.25,
        "stor_tinggi_a": p["min_storage"] * 0.75,
        "stor_tinggi_b": p["min_storage"] * 2.0,
        # Harga (cost): murah = di bawah max_harga, mahal = di atas
        "harga_murah_a": p["max_harga"] * 0.4,
        "harga_murah_b": p["max_harga"] * 1.1,
        "harga_mahal_a": p["max_harga"] * 0.9,
        "harga_mahal_b": p["max_harga"] * 1.8,
    }

# ─────────────────────────────────────────────────────────────────────
# INTI FUZZY TSUKAMOTO
# ─────────────────────────────────────────────────────────────────────
def tsukamoto_score(cpu, gpu, ram, storage, harga, params, bobot_norm):
    """
    Hitung skor kelayakan Fuzzy Tsukamoto [0..1].
    Rule base:
      IF cpu TINGGI   AND ... -> rekomendasi TINGGI  (z via invers_naik)
      IF cpu RENDAH   AND ... -> rekomendasi RENDAH  (z via invers_turun)
      IF harga MURAH  AND ... -> rekomendasi TINGGI
      IF harga MAHAL  AND ... -> rekomendasi RENDAH
    Setiap kriteria disumbangkan secara tertimbang.
    """
    p  = params
    w  = bobot_norm
    z0, z1 = 0.0, 1.0  # domain output

    rules = []

    # CPU (benefit)
    alpha_cpu_t = mf_naik(cpu,   p["cpu_tinggi_a"],  p["cpu_tinggi_b"])
    alpha_cpu_r = mf_turun(cpu,  p["cpu_rendah_a"],  p["cpu_rendah_b"])
    rules.append((w[0], alpha_cpu_t, invers_naik(alpha_cpu_t,  z0, z1)))
    rules.append((w[0], alpha_cpu_r, invers_turun(alpha_cpu_r, z0, z1)))

    # GPU (benefit)
    alpha_gpu_t = mf_naik(gpu,   p["gpu_tinggi_a"],  p["gpu_tinggi_b"])
    alpha_gpu_r = mf_turun(gpu,  p["gpu_rendah_a"],  p["gpu_rendah_b"])
    rules.append((w[1], alpha_gpu_t, invers_naik(alpha_gpu_t,  z0, z1)))
    rules.append((w[1], alpha_gpu_r, invers_turun(alpha_gpu_r, z0, z1)))

    # RAM (benefit)
    alpha_ram_t = mf_naik(ram,   p["ram_tinggi_a"],  p["ram_tinggi_b"])
    alpha_ram_r = mf_turun(ram,  p["ram_rendah_a"],  p["ram_rendah_b"])
    rules.append((w[2], alpha_ram_t, invers_naik(alpha_ram_t,  z0, z1)))
    rules.append((w[2], alpha_ram_r, invers_turun(alpha_ram_r, z0, z1)))

    # Storage (benefit)
    alpha_st_t = mf_naik(storage, p["stor_tinggi_a"], p["stor_tinggi_b"])
    alpha_st_r = mf_turun(storage, p["stor_rendah_a"],p["stor_rendah_b"])
    rules.append((w[3], alpha_st_t, invers_naik(alpha_st_t,  z0, z1)))
    rules.append((w[3], alpha_st_r, invers_turun(alpha_st_r, z0, z1)))

    # Harga (cost): murah -> rekomendasi TINGGI, mahal -> RENDAH
    alpha_h_m = mf_turun(harga, p["harga_murah_a"], p["harga_murah_b"])
    alpha_h_M = mf_naik(harga,  p["harga_mahal_a"], p["harga_mahal_b"])
    rules.append((w[4], alpha_h_m, invers_naik(alpha_h_m,  z0, z1)))
    rules.append((w[4], alpha_h_M, invers_turun(alpha_h_M, z0, z1)))

    num = sum(wi * a * z for wi, a, z in rules)
    den = sum(wi * a     for wi, a, z in rules)
    return num / den if den > 1e-9 else 0.0


def label_kelayakan(skor):
    if skor >= 0.75:  return "✅ Sangat Layak",   "#2ecc71"
    if skor >= 0.55:  return "🟡 Layak",          "#f1c40f"
    if skor >= 0.35:  return "🟠 Kurang Layak",   "#e67e22"
    return               "❌ Tidak Layak",         "#e74c3c"


def cek_minimum(cpu, gpu, ram, storage, harga, profil_key):
    """Cek apakah laptop memenuhi spek minimum hard-rule profil."""
    p = PROFIL[profil_key]
    hasil = {
        "CPU":     (cpu     >= p["min_cpu"],     cpu,     p["min_cpu"],     "multiScore"),
        "GPU":     (gpu     >= p["min_gpu"],     gpu,     p["min_gpu"],     "3DMark")     if p["min_gpu"] > 0 else None,
        "RAM":     (ram     >= p["min_ram"],     ram,     p["min_ram"],     "GB"),
        "Storage": (storage >= p["min_storage"], storage, p["min_storage"], "GB"),
        "Harga":   (harga   <= p["max_harga"],   harga,   p["max_harga"],   "USD (maks)"),
    }
    return {k: v for k, v in hasil.items() if v is not None}


# ─────────────────────────────────────────────────────────────────────
# LOAD & PERSIAPAN DATA
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_laptop = pd.read_csv('dataset/laptop.csv')
    df_claptop = df_laptop.drop(columns=["Laptop_ID", "Performance_Level"])

    df_cpu = pd.read_csv('dataset/CPU.csv')
    df_cpu["CPU"] = df_cpu["manufacturer"].astype(str) + " " + df_cpu["namaCPU"].astype(str)
    df_ccpu = df_cpu[["CPU", "multiScore"]].copy()
    df_ccpu["CPU_Clean"] = df_ccpu["CPU"].str.split().str[0:3].str.join(' ')
    df_claptop["CPU_Clean"] = df_claptop["CPU"].astype(str).str.split().str[0:3].str.join(' ')
    df_laptopcpu = pd.merge(df_claptop, df_ccpu, on="CPU_Clean", how="left")
    df_laptopcpu["multiScore"] = pd.to_numeric(df_laptopcpu["multiScore"], errors="coerce")
    df_laptopcpu = df_laptopcpu.rename(columns={"CPU_y": "CPU"})
    df_fixlaptopcpu = (df_laptopcpu.sort_values("multiScore", ascending=False)
                       .drop_duplicates("Model", keep="first"))

    df_gpu = pd.read_csv('dataset/GPU.csv')
    df_cgpu = df_gpu[["gpuName", "G3Dmark"]].rename(columns={"gpuName": "GPU", "G3Dmark": "3DMark"})
    df_claptop["GPU_Clean"] = df_claptop["GPU"].astype(str).str.split().str[0:1].str.join(' ')
    df_cgpu["GPU_Clean"] = df_cgpu["GPU"].str.split().str[0:1].str.join(' ')
    df_laptopgpu = pd.merge(df_claptop, df_cgpu, on="GPU_Clean", how="left")
    df_laptopgpu["3DMark"] = pd.to_numeric(df_laptopgpu["3DMark"], errors="coerce")
    df_laptopgpu = df_laptopgpu.rename(columns={"GPU_y": "GPU"})
    df_fixlaptopgpu = (df_laptopgpu.sort_values("3DMark", ascending=False)
                       .drop_duplicates("Model", keep="first"))

    s1 = pd.merge(df_claptop, df_fixlaptopcpu[["Model", "multiScore"]], on="Model", how="left")
    sf = pd.merge(s1, df_fixlaptopgpu[["Model", "3DMark"]], on="Model", how="left")
    df = sf.dropna(subset=["multiScore", "3DMark"])
    df = df.drop(columns=["CPU_Clean", "GPU_Clean", "Brand"], errors="ignore")
    df["Storage"]   = df["Storage"].apply(konversi_storage)
    df["RAM"]       = df["RAM"].apply(konversi_ram)
    df["Price_USD"] = df["Price_USD"].apply(float)
    df = df.reset_index(drop=True)
    return df, df_ccpu, df_cgpu

df_semua, df_ccpu, df_cgpu = load_data()
model_list = sorted(df_semua["Model"].unique().tolist())


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    page = st.selectbox("📂 Pilih Halaman", [
        "🏠 Beranda",
        "🔍 Cek Laptop",
        "📊 Ranking Semua Laptop",
        "📐 Transparansi Fuzzy",
        "📈 Visualisasi",
    ])
    st.markdown("___")
    st.markdown("**Metode:** Fuzzy Tsukamoto  \n**Data:** 113 laptop  \n**Kriteria:** CPU · GPU · RAM · Storage · Harga")


# ─────────────────────────────────────────────────────────────────────
# PAGE: BERANDA
# ─────────────────────────────────────────────────────────────────────
if page == "🏠 Beranda":
    st.title("💻 SPK Pemilihan Laptop – Fuzzy Tsukamoto")
    st.markdown(
        "Sistem ini menilai **kelayakan laptop** berdasarkan **profil kebutuhan pengguna** "
        "menggunakan metode **Fuzzy Tsukamoto**. Setiap profil memiliki batasan spek minimum "
        "dan bobot prioritas kriteria yang berbeda."
    )

    st.markdown("### 👤 Profil Pengguna yang Didukung")
    cols = st.columns(len(PROFIL))
    for i, (nama, info) in enumerate(PROFIL.items()):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"<h3 style='text-align:center'>{nama}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size:13px;color:#cbd5e1'>{info['deskripsi']}</p>", unsafe_allow_html=True)
                st.markdown("**Spek Minimum:**")
                st.markdown(f"- CPU: `≥ {info['min_cpu']:,}` multiScore")
                if info["min_gpu"] > 0:
                    st.markdown(f"- GPU: `≥ {info['min_gpu']:,}` 3DMark")
                else:
                    st.markdown("- GPU: *(tidak diwajibkan)*")
                st.markdown(f"- RAM: `≥ {info['min_ram']} GB`")
                st.markdown(f"- Storage: `≥ {info['min_storage']} GB`")
                st.markdown(f"- Harga: `≤ ${info['max_harga']:,}`")

    st.markdown("___")
    st.markdown("### ⚙️ Cara Kerja Fuzzy Tsukamoto")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        with st.container(border=True):
            st.markdown("**1️⃣ Fuzzifikasi**")
            st.markdown("Nilai spek laptop (crisp) diubah menjadi derajat keanggotaan μ ∈ [0,1] pada himpunan RENDAH dan TINGGI untuk setiap kriteria.")
    with col_b:
        with st.container(border=True):
            st.markdown("**2️⃣ Inferensi Rule**")
            st.markdown("Setiap kriteria mengaktifkan rule: IF spek TINGGI → output LAYAK. IF spek RENDAH → output TIDAK LAYAK.")
    with col_c:
        with st.container(border=True):
            st.markdown("**3️⃣ Defuzzifikasi**")
            st.markdown("Setiap rule menghasilkan nilai crisp via invers MF. Output akhir = rata-rata tertimbang semua rule.")

    st.latex(r"z^* = \frac{\sum_{i} w_i \cdot \alpha_i \cdot z_i}{\sum_{i} w_i \cdot \alpha_i} \in [0, 1]")
    st.markdown("Skor 0 = Tidak Layak sama sekali, Skor 1 = Sangat Layak untuk profil tersebut.")


# ─────────────────────────────────────────────────────────────────────
# PAGE: CEK LAPTOP
# ─────────────────────────────────────────────────────────────────────
elif page == "🔍 Cek Laptop":
    st.title("🔍 Cek Kelayakan Laptop")
    st.markdown("Pilih profil kebutuhanmu, lalu **pilih laptop dari dataset** atau **input spesifikasi manual** untuk dinilai.")

    # Pilih profil
    profil_key = st.selectbox("👤 Pilih Profil Kebutuhan", list(PROFIL.keys()))
    info_profil = PROFIL[profil_key]
    st.info(f"**{profil_key}** — {info_profil['deskripsi']}")

    with st.expander("📋 Lihat Spek Minimum & Bobot Profil Ini", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Spek Minimum (Hard Requirement):**")
            st.markdown(f"- CPU multiScore: `≥ {info_profil['min_cpu']:,}`")
            st.markdown(f"- GPU 3DMark: `≥ {info_profil['min_gpu']:,}`" if info_profil['min_gpu'] > 0 else "- GPU: *(tidak diwajibkan)*")
            st.markdown(f"- RAM: `≥ {info_profil['min_ram']} GB`")
            st.markdown(f"- Storage: `≥ {info_profil['min_storage']} GB`")
            st.markdown(f"- Harga: `≤ ${info_profil['max_harga']:,}`")
        with col2:
            nama_kriteria = ["CPU", "GPU", "RAM", "Storage", "Harga"]
            bobot_awal = info_profil["bobot"]
            bobot_norm = np.array(bobot_awal) / sum(bobot_awal)
            st.markdown("**Bobot Prioritas:**")
            for n, ba, bn in zip(nama_kriteria, bobot_awal, bobot_norm):
                bar = "█" * ba + "░" * (5 - ba)
                st.markdown(f"- {n}: `{bar}` ({ba}/5 → {bn:.2f})")

    st.markdown("---")

    # Input mode
    mode = st.radio("📥 Mode Input:", ["Pilih dari Dataset", "Input Manual"], horizontal=True)

    if mode == "Pilih dari Dataset":
        model_dipilih = st.selectbox("🖥️ Pilih Model Laptop", model_list)
        baris = df_semua[df_semua["Model"] == model_dipilih].iloc[0]
        cpu_val     = float(baris["multiScore"])
        gpu_val     = float(baris["3DMark"])
        ram_val     = float(baris["RAM"])
        storage_val = float(baris["Storage"])
        harga_val   = float(baris["Price_USD"])

        st.markdown("**Spesifikasi Laptop Terpilih:**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("CPU multiScore", f"{cpu_val:,.0f}")
        c2.metric("GPU 3DMark",     f"{gpu_val:,.0f}")
        c3.metric("RAM",            f"{ram_val:.0f} GB")
        c4.metric("Storage",        f"{storage_val:.0f} GB")
        c5.metric("Harga",          f"${harga_val:,.0f}")

    else:
        st.markdown("**Masukkan Spesifikasi Laptop Secara Manual:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            cpu_val     = st.number_input("CPU multiScore",  min_value=0,   max_value=100000, value=20000, step=500)
            gpu_val     = st.number_input("GPU 3DMark",      min_value=0,   max_value=50000,  value=15000, step=500)
        with c2:
            ram_val     = st.number_input("RAM (GB)",         min_value=2,   max_value=256,    value=16,    step=2)
            storage_val = st.number_input("Storage (GB)",     min_value=32,  max_value=8192,   value=512,   step=64)
        with c3:
            harga_val   = st.number_input("Harga (USD)",      min_value=100, max_value=10000,  value=1000,  step=50)

    st.markdown("---")

    if st.button("🚀 Hitung Kelayakan", type="primary", width='stretch'):
        params     = bangun_params(profil_key)
        bobot_norm = np.array(info_profil["bobot"], dtype=float) / sum(info_profil["bobot"])
        skor       = tsukamoto_score(cpu_val, gpu_val, ram_val, storage_val, harga_val, params, bobot_norm)
        label, warna = label_kelayakan(skor)

        st.markdown("## 🎯 Hasil Penilaian")

        # Skor utama
        col_skor, col_label = st.columns([1, 2])
        with col_skor:
            st.markdown(
                f"<div style='background:{warna}22;border:2px solid {warna};border-radius:12px;"
                f"padding:20px;text-align:center'>"
                f"<p style='font-size:14px;color:#aaa;margin:0'>Skor Fuzzy Tsukamoto</p>"
                f"<p style='font-size:52px;font-weight:bold;color:{warna};margin:0'>{skor:.3f}</p>"
                f"<p style='font-size:22px;margin:0'>{label}</p>"
                f"</div>", unsafe_allow_html=True
            )
        with col_label:
            st.markdown("**Interpretasi Skor:**")
            for batas, teks, w in [
                ("≥ 0.75", "✅ Sangat Layak   – Spek melampaui kebutuhan profil",       "#2ecc71"),
                ("≥ 0.55", "🟡 Layak          – Memenuhi kebutuhan utama profil",        "#f1c40f"),
                ("≥ 0.35", "🟠 Kurang Layak   – Beberapa spek di bawah standar",         "#e67e22"),
                ("< 0.35", "❌ Tidak Layak    – Spek tidak mencukupi untuk profil ini",  "#e74c3c"),
            ]:
                bg = f"{w}33" if label.split()[0] == teks.split()[0] else "transparent"
                st.markdown(
                    f"<div style='background:{bg};padding:4px 10px;border-radius:6px;margin:3px 0'>"
                    f"<code>{batas}</code>  {teks}</div>",
                    unsafe_allow_html=True
                )

        # Cek spek minimum (hard rule)
        st.markdown("### 📋 Pengecekan Spek Minimum")
        hasil_min = cek_minimum(cpu_val, gpu_val, ram_val, storage_val, harga_val, profil_key)
        semua_lolos = all(v[0] for v in hasil_min.values())
        if semua_lolos:
            st.success("✅ Laptop memenuhi **semua** spek minimum untuk profil ini.")
        else:
            st.warning("⚠️ Laptop **tidak memenuhi** sebagian spek minimum. Detail di bawah.")

        cols_min = st.columns(len(hasil_min))
        for i, (nama, (lolos, nilai, minimum, satuan)) in enumerate(hasil_min.items()):
            ikon  = "✅" if lolos else "❌"
            warna_card = "#1e3a1e" if lolos else "#3a1e1e"
            warna_teks = "#2ecc71" if lolos else "#e74c3c"
            label_harga = "≤" if nama == "Harga" else "≥"
            with cols_min[i]:
                st.markdown(
                    f"<div style='background:{warna_card};border-radius:10px;padding:12px;text-align:center'>"
                    f"<p style='font-size:22px;margin:0'>{ikon}</p>"
                    f"<p style='font-weight:bold;color:{warna_teks};margin:4px 0'>{nama}</p>"
                    f"<p style='font-size:13px;margin:0;color:#ddd'>Nilai: <b>{nilai:,.0f}</b> {satuan}</p>"
                    f"<p style='font-size:12px;color:#aaa;margin:0'>Min {label_harga} {minimum:,.0f}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Breakdown derajat keanggotaan per kriteria
        st.markdown("### 🔬 Detail Fuzzifikasi per Kriteria")
        st.caption("Derajat keanggotaan (μ) dan kontribusi skor masing-masing kriteria:")

        p = params
        detail = [
            ("CPU",     cpu_val,     "multiScore", p["cpu_tinggi_a"],  p["cpu_tinggi_b"],  p["cpu_rendah_a"],  p["cpu_rendah_b"],  bobot_norm[0], "Benefit"),
            ("GPU",     gpu_val,     "3DMark",     p["gpu_tinggi_a"],  p["gpu_tinggi_b"],  p["gpu_rendah_a"],  p["gpu_rendah_b"],  bobot_norm[1], "Benefit"),
            ("RAM",     ram_val,     "GB",         p["ram_tinggi_a"],  p["ram_tinggi_b"],  p["ram_rendah_a"],  p["ram_rendah_b"],  bobot_norm[2], "Benefit"),
            ("Storage", storage_val, "GB",         p["stor_tinggi_a"], p["stor_tinggi_b"], p["stor_rendah_a"], p["stor_rendah_b"], bobot_norm[3], "Benefit"),
            ("Harga",   harga_val,   "USD",        p["harga_mahal_a"], p["harga_mahal_b"], p["harga_murah_a"], p["harga_murah_b"], bobot_norm[4], "Cost"),
        ]

        rows_detail = []
        for nama, val, satuan, ta, tb, ra, rb, w, sifat in detail:
            if sifat == "Benefit":
                mu_t = mf_naik(val,  ta, tb)
                mu_r = mf_turun(val, ra, rb)
                label_t, label_r = "TINGGI", "RENDAH"
            else:
                mu_t = mf_naik(val,  ta, tb)
                mu_r = mf_turun(val, ra, rb)
                label_t, label_r = "MAHAL", "MURAH"
            z_t = invers_naik(mu_t,  0, 1)
            z_r = invers_turun(mu_r, 0, 1)
            kontrib = (w * mu_t * z_t + w * mu_r * z_r) / max(w * mu_t + w * mu_r, 1e-9)
            rows_detail.append({
                "Kriteria":        nama,
                "Nilai Input":     f"{val:,.0f} {satuan}",
                "Sifat":           sifat,
                f"μ {label_r}":    round(mu_r, 4),
                f"μ {label_t}":    round(mu_t, 4),
                "Bobot (w)":       round(w, 4),
                "Skor Kontribusi": round(kontrib, 4),
            })

        df_detail = pd.DataFrame(rows_detail).set_index("Kriteria")
        st.dataframe(df_detail, width='stretch')

        # Plot kontribusi
        fig, ax = plt.subplots(figsize=(10, 3.5))
        krits = [r["Kriteria"] for r in rows_detail]
        kontribs = [r["Skor Kontribusi"] for r in rows_detail]
        bar_colors = ["#2ecc71" if k >= skor else "#e74c3c" for k in kontribs]
        bars = ax.bar(krits, kontribs, color=bar_colors, edgecolor='white', linewidth=1.2)
        ax.axhline(skor, color='#f1c40f', linewidth=2, linestyle='--', label=f"Skor Total: {skor:.3f}")
        ax.axhline(0.55, color='#aaa',    linewidth=1, linestyle=':',  label="Ambang Layak (0.55)")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha='center', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Skor Kontribusi")
        ax.set_title("Kontribusi Skor per Kriteria vs Skor Total", fontweight='bold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')

        # Rekomendasi laptop serupa yang lebih layak (jika tidak lolos)
        if skor < 0.55:
            st.markdown("### 💡 Rekomendasi Laptop Alternatif yang Lebih Layak")
            st.caption(f"Laptop dari dataset yang **layak** untuk profil **{profil_key}** berdasarkan Fuzzy Tsukamoto:")
            df_rec = df_semua.copy()
            df_rec["Skor"] = df_rec.apply(
                lambda r: tsukamoto_score(r["multiScore"], r["3DMark"], r["RAM"],
                                          r["Storage"], r["Price_USD"], params, bobot_norm), axis=1
            )
            df_rec = df_rec[df_rec["Skor"] >= 0.55].sort_values("Skor", ascending=False).head(5)
            if len(df_rec) > 0:
                df_rec_show = df_rec[["Model", "RAM", "Storage", "Price_USD", "multiScore", "3DMark", "Skor"]].copy()
                df_rec_show.columns = ["Model","RAM(GB)","Storage(GB)","Harga(USD)","CPU Score","GPU Score","Skor Tsukamoto"]
                df_rec_show.index = range(1, len(df_rec_show)+1)
                st.dataframe(df_rec_show, width='stretch')
            else:
                st.info("Tidak ada laptop di dataset yang memenuhi ambang kelayakan untuk profil ini.")


# ─────────────────────────────────────────────────────────────────────
# PAGE: RANKING SEMUA LAPTOP
# ─────────────────────────────────────────────────────────────────────
elif page == "📊 Ranking Semua Laptop":
    st.title("📊 Ranking Semua Laptop per Profil")
    profil_key = st.selectbox("👤 Pilih Profil", list(PROFIL.keys()))

    params     = bangun_params(profil_key)
    bobot_norm = np.array(PROFIL[profil_key]["bobot"], dtype=float) / sum(PROFIL[profil_key]["bobot"])

    df_rank = df_semua.copy()
    df_rank["Skor_Tsukamoto"] = df_rank.apply(
        lambda r: tsukamoto_score(r["multiScore"], r["3DMark"], r["RAM"],
                                  r["Storage"], r["Price_USD"], params, bobot_norm), axis=1
    )
    df_rank = df_rank.sort_values("Skor_Tsukamoto", ascending=False).reset_index(drop=True)
    df_rank.index = df_rank.index + 1
    df_rank["Kelayakan"] = df_rank["Skor_Tsukamoto"].apply(lambda s: label_kelayakan(s)[0])

    st.info(f"Menampilkan ranking {len(df_rank)} laptop untuk profil **{profil_key}**")

    # Metrik ringkasan
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Laptop Sangat Layak", len(df_rank[df_rank["Skor_Tsukamoto"] >= 0.75]))
    c2.metric("Laptop Layak",        len(df_rank[(df_rank["Skor_Tsukamoto"] >= 0.55) & (df_rank["Skor_Tsukamoto"] < 0.75)]))
    c3.metric("Kurang Layak",        len(df_rank[(df_rank["Skor_Tsukamoto"] >= 0.35) & (df_rank["Skor_Tsukamoto"] < 0.55)]))
    c4.metric("Tidak Layak",         len(df_rank[df_rank["Skor_Tsukamoto"] < 0.35]))

    # Filter
    filter_kel = st.multiselect("Filter Kelayakan:", ["✅ Sangat Layak","🟡 Layak","🟠 Kurang Layak","❌ Tidak Layak"],
                                 default=["✅ Sangat Layak","🟡 Layak"])
    df_filtered = df_rank[df_rank["Kelayakan"].isin(filter_kel)] if filter_kel else df_rank

    kolom_show = ["Model","RAM","Storage","Price_USD","multiScore","3DMark","Skor_Tsukamoto","Kelayakan"]
    df_show = df_filtered[kolom_show].copy()
    df_show.columns = ["Model","RAM(GB)","Storage(GB)","Harga(USD)","CPU Score","GPU Score","Skor Tsukamoto","Kelayakan"]
    st.dataframe(df_show, width='stretch')

    # Top 10 bar chart
    st.markdown("### 📊 Top 10 Laptop Terbaik")
    df_top10 = df_rank.head(10)
    fig, ax = plt.subplots(figsize=(12, 5))
    skor_colors = [
        "#2ecc71" if s >= 0.75 else "#f1c40f" if s >= 0.55 else "#e67e22" if s >= 0.35 else "#e74c3c"
        for s in df_top10["Skor_Tsukamoto"]
    ]
    bars = ax.barh(df_top10["Model"][::-1], df_top10["Skor_Tsukamoto"][::-1],
                   color=skor_colors[::-1], edgecolor='white', linewidth=0.8)
    ax.axvline(0.55, color='#f1c40f', linewidth=1.5, linestyle='--', label='Ambang Layak (0.55)')
    ax.axvline(0.75, color='#2ecc71', linewidth=1.5, linestyle='--', label='Ambang Sangat Layak (0.75)')
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}",
                va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Skor Fuzzy Tsukamoto")
    ax.set_title(f"Top 10 Laptop – Profil {profil_key}", fontweight='bold')
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, width='stretch')


# ─────────────────────────────────────────────────────────────────────
# PAGE: TRANSPARANSI FUZZY
# ─────────────────────────────────────────────────────────────────────
elif page == "📐 Transparansi Fuzzy":
    st.title("📐 Transparansi Perhitungan Fuzzy Tsukamoto")
    profil_key = st.selectbox("👤 Profil:", list(PROFIL.keys()))
    params     = bangun_params(profil_key)
    bobot_norm = np.array(PROFIL[profil_key]["bobot"], dtype=float) / sum(PROFIL[profil_key]["bobot"])

    with st.expander("📋 Tabel Parameter Fungsi Keanggotaan", expanded=True):
        st.markdown("Batas fungsi keanggotaan diturunkan dari spek minimum profil. "
                    "Fungsi **RENDAH/MURAH** = bahu turun, **TINGGI/MAHAL** = bahu naik.")
        df_par = pd.DataFrame({
            "Kriteria":        ["CPU","GPU","RAM","Storage","Harga"],
            "Sifat":           ["Benefit","Benefit","Benefit","Benefit","Cost"],
            "a RENDAH/MURAH":  [params["cpu_rendah_a"],  params["gpu_rendah_a"],  params["ram_rendah_a"],  params["stor_rendah_a"],  params["harga_murah_a"]],
            "b RENDAH/MURAH":  [params["cpu_rendah_b"],  params["gpu_rendah_b"],  params["ram_rendah_b"],  params["stor_rendah_b"],  params["harga_murah_b"]],
            "a TINGGI/MAHAL":  [params["cpu_tinggi_a"],  params["gpu_tinggi_a"],  params["ram_tinggi_a"],  params["stor_tinggi_a"],  params["harga_mahal_a"]],
            "b TINGGI/MAHAL":  [params["cpu_tinggi_b"],  params["gpu_tinggi_b"],  params["ram_tinggi_b"],  params["stor_tinggi_b"],  params["harga_mahal_b"]],
            "Bobot (w)":       np.round(bobot_norm, 4),
        })
        df_par.index = df_par.index + 1
        st.dataframe(df_par, width='stretch')

    with st.expander("📏 Rule Base Fuzzy Tsukamoto", expanded=True):
        st.markdown("**10 Rule aktif** (2 rule per kriteria × 5 kriteria):")
        rules_text = [
            ("R1",  "CPU",     "TINGGI",  "→ output TINGGI (z via invers_naik)",  "Benefit"),
            ("R2",  "CPU",     "RENDAH",  "→ output RENDAH (z via invers_turun)", "Benefit"),
            ("R3",  "GPU",     "TINGGI",  "→ output TINGGI",                       "Benefit"),
            ("R4",  "GPU",     "RENDAH",  "→ output RENDAH",                       "Benefit"),
            ("R5",  "RAM",     "TINGGI",  "→ output TINGGI",                       "Benefit"),
            ("R6",  "RAM",     "RENDAH",  "→ output RENDAH",                       "Benefit"),
            ("R7",  "Storage", "TINGGI",  "→ output TINGGI",                       "Benefit"),
            ("R8",  "Storage", "RENDAH",  "→ output RENDAH",                       "Benefit"),
            ("R9",  "Harga",   "MURAH",   "→ output TINGGI (harga murah = bagus)", "Cost"),
            ("R10", "Harga",   "MAHAL",   "→ output RENDAH",                       "Cost"),
        ]
        df_rules = pd.DataFrame(rules_text, columns=["Rule","Kriteria","Himpunan","Konsekuen","Sifat"])
        df_rules.index = df_rules.index
        st.dataframe(df_rules, width='stretch')

    with st.expander("🔢 Tabel Derajat Keanggotaan Semua Laptop", expanded=False):
        st.caption("Derajat μ setiap laptop pada himpunan fuzzy masing-masing kriteria:")
        rows_mu = []
        for _, r in df_semua.iterrows():
            rows_mu.append({
                "Model":           r["Model"],
                "μ CPU Rendah":    round(mf_turun(r["multiScore"], params["cpu_rendah_a"],  params["cpu_rendah_b"]),  4),
                "μ CPU Tinggi":    round(mf_naik( r["multiScore"], params["cpu_tinggi_a"],  params["cpu_tinggi_b"]),  4),
                "μ GPU Rendah":    round(mf_turun(r["3DMark"],     params["gpu_rendah_a"],  params["gpu_rendah_b"]),  4),
                "μ GPU Tinggi":    round(mf_naik( r["3DMark"],     params["gpu_tinggi_a"],  params["gpu_tinggi_b"]),  4),
                "μ RAM Rendah":    round(mf_turun(r["RAM"],        params["ram_rendah_a"],  params["ram_rendah_b"]),  4),
                "μ RAM Tinggi":    round(mf_naik( r["RAM"],        params["ram_tinggi_a"],  params["ram_tinggi_b"]),  4),
                "μ Stor Rendah":   round(mf_turun(r["Storage"],    params["stor_rendah_a"], params["stor_rendah_b"]), 4),
                "μ Stor Tinggi":   round(mf_naik( r["Storage"],    params["stor_tinggi_a"], params["stor_tinggi_b"]), 4),
                "μ Harga Murah":   round(mf_turun(r["Price_USD"],  params["harga_murah_a"], params["harga_murah_b"]), 4),
                "μ Harga Mahal":   round(mf_naik( r["Price_USD"],  params["harga_mahal_a"], params["harga_mahal_b"]), 4),
            })
        df_mu = pd.DataFrame(rows_mu).set_index("Model")
        st.dataframe(df_mu, width='stretch')


# ─────────────────────────────────────────────────────────────────────
# PAGE: VISUALISASI
# ─────────────────────────────────────────────────────────────────────
elif page == "📈 Visualisasi":
    st.title("📈 Visualisasi – Fuzzy Tsukamoto")
    profil_key = st.selectbox("👤 Profil:", list(PROFIL.keys()))
    params     = bangun_params(profil_key)
    bobot_norm = np.array(PROFIL[profil_key]["bobot"], dtype=float) / sum(PROFIL[profil_key]["bobot"])

    df_vis = df_semua.copy()
    df_vis["Skor"] = df_vis.apply(
        lambda r: tsukamoto_score(r["multiScore"], r["3DMark"], r["RAM"],
                                  r["Storage"], r["Price_USD"], params, bobot_norm), axis=1
    )
    df_vis = df_vis.sort_values("Skor", ascending=False).reset_index(drop=True)

    # Grafik 1: Distribusi Skor
    st.subheader("📊 Distribusi Skor Kelayakan Semua Laptop")
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    xs = df_vis.index + 1
    bar_cols = [
        "#2ecc71" if s >= 0.75 else "#f1c40f" if s >= 0.55 else "#e67e22" if s >= 0.35 else "#e74c3c"
        for s in df_vis["Skor"]
    ]
    ax1.bar(xs, df_vis["Skor"], color=bar_cols, width=1.0, edgecolor='none')
    ax1.axhline(0.75, color='#2ecc71', linewidth=1.5, linestyle='--', label='Sangat Layak (0.75)')
    ax1.axhline(0.55, color='#f1c40f', linewidth=1.5, linestyle='--', label='Layak (0.55)')
    ax1.axhline(0.35, color='#e67e22', linewidth=1.5, linestyle='--', label='Kurang Layak (0.35)')
    ax1.set_xlabel("Laptop (urut ranking)")
    ax1.set_ylabel("Skor Tsukamoto")
    ax1.set_title(f"Distribusi Skor – Profil {profil_key}", fontweight='bold')
    patch_list = [
        mpatches.Patch(color='#2ecc71', label='Sangat Layak'),
        mpatches.Patch(color='#f1c40f', label='Layak'),
        mpatches.Patch(color='#e67e22', label='Kurang Layak'),
        mpatches.Patch(color='#e74c3c', label='Tidak Layak'),
    ]
    ax1.legend(handles=patch_list, loc='upper right', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig1, width='stretch')

    # Grafik 2: Fungsi Keanggotaan (5 panel)
    st.subheader("📐 Fungsi Keanggotaan Fuzzy per Kriteria")
    p = params
    kriteria_mf = [
        ("CPU (multiScore)", p["cpu_rendah_a"],  p["cpu_rendah_b"],  p["cpu_tinggi_a"],  p["cpu_tinggi_b"],  "RENDAH","TINGGI"),
        ("GPU (3DMark)",     p["gpu_rendah_a"],  p["gpu_rendah_b"],  p["gpu_tinggi_a"],  p["gpu_tinggi_b"],  "RENDAH","TINGGI"),
        ("RAM (GB)",         p["ram_rendah_a"],  p["ram_rendah_b"],  p["ram_tinggi_a"],  p["ram_tinggi_b"],  "RENDAH","TINGGI"),
        ("Storage (GB)",     p["stor_rendah_a"], p["stor_rendah_b"], p["stor_tinggi_a"], p["stor_tinggi_b"], "RENDAH","TINGGI"),
        ("Harga (USD)",      p["harga_murah_a"], p["harga_murah_b"], p["harga_mahal_a"], p["harga_mahal_b"], "MURAH","MAHAL"),
    ]

    fig2, axes = plt.subplots(1, 5, figsize=(20, 4))
    for idx, (label, ra, rb, ta, tb, ll, lh) in enumerate(kriteria_mf):
        ax = axes[idx]
        xmin = min(ra, ta) * 0.85
        xmax = max(rb, tb) * 1.15
        xs = np.linspace(xmin, xmax, 400)
        mu_r = [mf_turun(x, ra, rb) for x in xs]
        mu_t = [mf_naik(x, ta, tb) for x in xs]
        ax.plot(xs, mu_r, color='#e74c3c', linewidth=2.5, label=ll)
        ax.plot(xs, mu_t, color='#2ecc71', linewidth=2.5, label=lh)
        ax.fill_between(xs, mu_r, alpha=0.12, color='#e74c3c')
        ax.fill_between(xs, mu_t, alpha=0.12, color='#2ecc71')
        # garis spek minimum
        sp_min = [p["cpu_rendah_a"], p["gpu_rendah_a"], p["ram_rendah_a"], p["stor_rendah_a"], p["harga_murah_b"]]
        ax.axvline(sp_min[idx], color='#f1c40f', linewidth=1.5, linestyle=':', label='Spek Min')
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel("Nilai", fontsize=8)
        ax.set_ylabel("μ", fontsize=9)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(f"Fungsi Keanggotaan – Profil {profil_key}", fontsize=12, fontweight='bold', y=1.03)
    plt.tight_layout()
    st.pyplot(fig2, width='stretch')

    # Grafik 3: Scatter Harga vs Skor
    st.subheader("💰 Scatter: Harga vs Skor Kelayakan")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    scatter_cols = [
        "#2ecc71" if s >= 0.75 else "#f1c40f" if s >= 0.55 else "#e67e22" if s >= 0.35 else "#e74c3c"
        for s in df_vis["Skor"]
    ]
    ax3.scatter(df_vis["Price_USD"], df_vis["Skor"], c=scatter_cols, s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax3.axhline(0.55, color='#f1c40f', linewidth=1.5, linestyle='--', alpha=0.8, label='Ambang Layak')
    ax3.axvline(PROFIL[profil_key]["max_harga"], color='#aaa', linewidth=1.5, linestyle=':', label=f"Maks Harga (${PROFIL[profil_key]['max_harga']:,})")
    ax3.set_xlabel("Harga (USD)", fontsize=11)
    ax3.set_ylabel("Skor Tsukamoto", fontsize=11)
    ax3.set_title("Korelasi Harga vs Kelayakan", fontweight='bold')
    for patch, lbl in [("#2ecc71","Sangat Layak"),("#f1c40f","Layak"),("#e67e22","Kurang Layak"),("#e74c3c","Tidak Layak")]:
        ax3.scatter([], [], c=patch, s=60, label=lbl)
    ax3.legend(fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3, width='stretch')
