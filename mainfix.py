import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pemilihan Laptop - Fuzzy Tsukamoto", layout="wide")

# 1. FUNGSI KONVERSI DATA & FUZZY
def konversi_storage(x):
    text = str(x).strip().upper()
    angka_text = "".join([char for char in text if char.isdigit()])
    if not angka_text : return 0
    angka = float(angka_text)
    if "TB" in text: return angka * 1024
    return angka

def konversi_ram(x):  
    text = str(x).strip().upper()
    angka_text = "".join([char for char in text if char.isdigit()])
    if not angka_text: return 0 
    return float(angka_text)
    
def konversi_Price(x):
    return float(x)

@st.cache_data
def load_data(dataset):
    return pd.read_csv(dataset)

# Fungsi Kurva Keanggotaan
def kurva_turun(x, batas_bawah, batas_atas):
    if x <= batas_bawah: return 1.0
    if x >= batas_atas: return 0.0
    return (batas_atas - x) / (batas_atas - batas_bawah)

def kurva_naik(x, batas_bawah, batas_atas):
    if x <= batas_bawah: return 0.0
    if x >= batas_atas: return 1.0
    return (x - batas_bawah) / (batas_atas - batas_bawah)

def z_turun(alpha, batas_bawah=0, batas_atas=100):
    return batas_atas - (alpha * (batas_atas - batas_bawah))

def z_naik(alpha, batas_bawah=0, batas_atas=100):
    return batas_bawah + (alpha * (batas_atas - batas_bawah))

# Fungsi Hitung Skor Tsukamoto Akhir
def hitung_tsukamoto(cpu_score, gpu_score, ram, storage, harga):
    # Fuzzifikasi
    cpu_rendah = kurva_turun(cpu_score, 8000, 30000)
    cpu_tinggi = kurva_naik(cpu_score, 8000, 30000)
    
    gpu_rendah = kurva_turun(gpu_score, 10000, 30000)
    gpu_tinggi = kurva_naik(gpu_score, 10000, 30000)
    
    ram_kecil = kurva_turun(ram, 8, 64)
    ram_besar = kurva_naik(ram, 8, 64)
    
    storage_kecil = kurva_turun(storage, 256, 2048)
    storage_besar = kurva_naik(storage, 256, 2048)
    
    harga_murah = kurva_turun(harga, 200, 3000)
    harga_mahal = kurva_naik(harga, 200, 3000)
    
    # Inferensi (6 Aturan)
    a1 = min(cpu_tinggi, gpu_tinggi, ram_besar, storage_besar, harga_murah); z1 = z_naik(a1, 0, 100)
    a2 = gpu_rendah; z2 = z_turun(a2, 0, 100)
    a3 = harga_mahal; z3 = z_turun(a3, 0, 100)
    a4 = min(cpu_rendah, gpu_tinggi, harga_murah); z4 = z_naik(a4, 0, 75) 
    a5 = ram_kecil; z5 = z_turun(a5, 0, 100)
    a6 = storage_kecil; z6 = z_turun(a6, 0, 100)
    
    # Defuzzifikasi (Weighted Average)
    total_alpha = a1 + a2 + a3 + a4 + a5 + a6
    if total_alpha == 0: return 0
    return ((a1 * z1) + (a2 * z2) + (a3 * z3) + (a4 * z4) + (a5 * z5) + (a6 * z6)) / total_alpha

# 2. SIDEBAR
with st.sidebar:
    page = st.selectbox("Pilih Halaman",[
        "Data RAW",
        "Data CPU",
        "Data GPU",
        "Fuzzy Tsukamoto",
    ])

# 3. PAGE: DATA RAW
if page == "Data RAW" : 
    st.title("💻 Sistem Pendukung Keputusan Pemilihan Laptop")
    st.markdown("Aplikasi ini menggunakan **Metode Fuzzy Tsukamoto** untuk merekomendasikan laptop terbaik berdasarkan 5 kriteria.")

    st.markdown("### 📋 Kriteria Evaluasi Sistem")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        with st.container(border=True):
            st.markdown("<b style='color: #2ecc71; font-size: 17px;'>💻 CPU</b>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 13px; margin-top: 8px; color: #cbd5e1;'>Diukur dari nilai <i>multiScore</i> benchmark CPU.</p>", unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.markdown("<b style='color: #2ecc71; font-size: 17px;'>🎮 GPU</b>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 13px; margin-top: 8px; color: #cbd5e1;'>Diukur dari skor <i>3DMark</i> grafis.</p>", unsafe_allow_html=True)

    with col3:
        with st.container(border=True):
            st.markdown("<b style='color: #2ecc71; font-size: 17px;'>⚡ RAM</b>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 13px; margin-top: 8px; color: #cbd5e1;'>Kapasitas memori utama laptop (GB).</p>", unsafe_allow_html=True)

    with col4:
        with st.container(border=True):
            st.markdown("<b style='color: #2ecc71; font-size: 17px;'>💾 Storage</b>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 13px; margin-top: 8px; color: #cbd5e1;'>Kapasitas penyimpanan data (GB).</p>", unsafe_allow_html=True)

    with col5:
        with st.container(border=True):
            st.markdown("<b style='color: #2ecc71; font-size: 17px;'>💰 Harga</b>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 13px; margin-top: 8px; color: #cbd5e1;'>Harga laptop (USD).</p>", unsafe_allow_html=True)

    st.write("Data Alternatif Laptop")
    try:
        df_laptop = load_data('dataset/laptop.csv')
        df_claptop = df_laptop.drop(columns=["Laptop_ID", "Performance_Level"], errors='ignore')
        df_claptop.index = df_claptop.index + 1
        st.dataframe(df_claptop)
        st.session_state["df_rawlaptop"] = df_claptop
    except Exception as e:
        st.error(f"Gagal memuat dataset laptop: {e}")

# 4. PAGE: DATA CPU
elif page == "Data CPU" :
    if "df_rawlaptop" in st.session_state:
        df_claptop = st.session_state["df_rawlaptop"].copy()
        
        st.title("🖥️ Analisis & Pemrosesan Data CPU")
        
        try:
            df_cpu = load_data('dataset/CPU.csv')
            df_cpu["CPU"] = df_cpu["manufacturer"].astype(str) + " " + df_cpu["namaCPU"].astype(str)
            df_ccpu = df_cpu[["CPU", "multiScore"]].copy()

            df_claptop["CPU_Clean"] = df_claptop["CPU"].astype(str).str.split().str[0:3].str.join(' ')
            df_ccpu["CPU_Clean"] = df_cpu["CPU"].astype(str).str.split().str[0:3].str.join(' ')
            
            df_laptopcpu = pd.merge(df_claptop, df_ccpu, on="CPU_Clean", how="left")
            
            df_dislaptopcpu = df_laptopcpu.drop(columns=["CPU_Clean","Brand","RAM","GPU","Price_USD","Storage", "CPU_x"], errors='ignore')
            if 'CPU_y' in df_dislaptopcpu.columns: df_dislaptopcpu = df_dislaptopcpu.rename(columns={"CPU_y": "CPU"})
                
            df_dislaptopcpu["multiScore"] = pd.to_numeric(df_dislaptopcpu["multiScore"], errors="coerce")
            df_fixlaptopcpu = df_dislaptopcpu.sort_values(by="multiScore", ascending=False).drop_duplicates(subset="Model", keep="first").reset_index(drop=True)
            df_fixlaptopcpu.index = df_fixlaptopcpu.index + 1
            st.session_state["df_fixlaptopcpu"] = df_fixlaptopcpu

            tab1, tab2 = st.tabs(["📊 Hasil Pemetaan Laptop & CPU", "📂 Dataset Raw CPU Benchmark"])
            with tab1:
                st.dataframe(df_fixlaptopcpu, width='stretch')
            with tab2:
                st.dataframe(df_ccpu.drop(columns=["CPU_Clean"], errors="ignore"), width='stretch')
        except Exception as e:
            st.error(f"Gagal memuat dataset CPU: {e}")
    else:
        st.warning("Silakan buka halaman Data RAW terlebih dahulu.")

# 5. PAGE: DATA GPU
elif page == "Data GPU":
    if "df_rawlaptop" in st.session_state:
        df_claptop = st.session_state["df_rawlaptop"].copy()

        st.title("🎮 Analisis & Pemrosesan Data GPU")
        
        try:
            df_gpu = load_data('dataset/GPU.csv')
            df_cgpu = df_gpu.rename(columns={"gpuName": "GPU", "G3Dmark" : "3DMark"})[["GPU", "3DMark"]]

            df_claptop["GPU_Clean"] = df_claptop["GPU"].astype(str).str.split().str[0:1].str.join(' ')
            df_cgpu["GPU_Clean"] = df_cgpu["GPU"].astype(str).str.split().str[0:1].str.join(' ')
            
            df_laptopgpu = pd.merge(df_claptop, df_cgpu, on="GPU_Clean", how="left")
            
            df_dislaptopgpu = df_laptopgpu.drop(columns=["GPU_Clean","Brand","RAM","CPU","Price_USD","Storage","CPU_Clean", "GPU_x"], errors='ignore')
            if 'GPU_y' in df_dislaptopgpu.columns: df_dislaptopgpu = df_dislaptopgpu.rename(columns={"GPU_y": "GPU"})
                
            df_dislaptopgpu["3DMark"] = pd.to_numeric(df_dislaptopgpu["3DMark"], errors="coerce")
            df_fixlaptopgpu = df_dislaptopgpu.sort_values(by="3DMark", ascending=False).drop_duplicates(subset="Model", keep="first").reset_index(drop=True)
            df_fixlaptopgpu.index = df_fixlaptopgpu.index + 1
            st.session_state["df_fixlaptopgpu"] = df_fixlaptopgpu

            tab1, tab2 = st.tabs(["📊 Hasil Pemetaan Laptop & GPU", "📂 Dataset Raw GPU Benchmark"])
            with tab1:
                st.dataframe(df_fixlaptopgpu, width='stretch')
            with tab2:
                st.dataframe(df_cgpu.drop(columns=["GPU_Clean"], errors="ignore"), width='stretch')
        except Exception as e:
            st.error(f"Gagal memuat dataset GPU: {e}")
    else:
        st.warning("Silakan buka halaman Data RAW terlebih dahulu.")

# FUZZY TSUKAMOTO
elif page == "Fuzzy Tsukamoto" :
    if "df_rawlaptop" in st.session_state and "df_fixlaptopcpu" in st.session_state and "df_fixlaptopgpu" in st.session_state:
        df_claptop = st.session_state["df_rawlaptop"].copy()
        df_fixlaptopcpu = st.session_state["df_fixlaptopcpu"].copy()
        df_fixlaptopgpu = st.session_state["df_fixlaptopgpu"].copy()

        st.title("🧮 Proses Perhitungan Metode Fuzzy Tsukamoto")

        with st.expander("🛠️ Penggabungan & Transformasi Tipe Data", expanded=False):
            df_step1 = pd.merge(df_claptop, df_fixlaptopcpu[["Model", "multiScore"]], on="Model", how="left")
            df_datafinal = pd.merge(df_step1, df_fixlaptopgpu[["Model", "3DMark"]], on="Model", how="left").dropna()

            df_datafinal["Storage_GB"] = df_datafinal["Storage"].apply(konversi_storage)
            df_datafinal["RAM_GB"] = df_datafinal["RAM"].apply(konversi_ram)
            df_datafinal["Price_USD"] = df_datafinal["Price_USD"].apply(konversi_Price)
            st.dataframe(df_datafinal)

        with st.expander("⚖️ Langkah 1: Penentuan Rentang & Visualisasi Himpunan", expanded=False):
            st.markdown("#### **Tabel Penentuan Rentang (Batas Himpunan)**")
            df_rentang = pd.DataFrame({
                "Kriteria": ["CPU (MultiScore)", "GPU (3DMark)", "RAM (GB)", "Storage (GB)", "Harga (USD)", "Skor Kelayakan"],
                "Batas Bawah (Nilai Minimum)": ["8.000 (Rendah)", "10.000 (Rendah)", "8 (Kecil)", "256 (Kecil)", "200 (Murah)","0 (Rendah)"],
                "Batas Atas (Nilai Maksimum)": ["30.000 (Tinggi)", "30.000 (Tinggi)", "64 (Besar)", "2.048 (Besar)", "3.000 (Mahal)","100 (Tinggi)"]
            })
            st.table(df_rentang.set_index("Kriteria"))
            
            st.markdown("---")
            st.markdown("#### **Grafik Pemetaan Fungsi Keanggotaan**")

            def plot_turun(x, a, b): return np.clip((b - x) / (b - a), 0, 1)
            def plot_naik(x, a, b): return np.clip((x - a) / (b - a), 0, 1)

            x_cpu = np.linspace(0, 35000, 500)
            x_gpu = np.linspace(0, 35000, 500)
            x_ram = np.linspace(0, 72, 500)
            x_storage = np.linspace(0, 2100, 500)
            x_harga = np.linspace(0, 3500, 500)
            x_out = np.linspace(0, 100, 500)

            fig, grafik = plt.subplots(3, 2, figsize=(14, 15))

            grafik[0,0].plot(x_cpu, plot_turun(x_cpu, 8000, 30000), 'b', label='Rendah')
            grafik[0,0].plot(x_cpu, plot_naik(x_cpu, 8000, 30000), 'r', label='Tinggi')
            grafik[0,0].set_title('Kriteria: CPU MultiScore (8000 - 30000)')
            grafik[0,0].legend(); grafik[0,0].grid(True, linestyle='--', alpha=0.6)

            grafik[0,1].plot(x_gpu, plot_turun(x_gpu, 10000, 30000), 'b', label='Rendah')
            grafik[0,1].plot(x_gpu, plot_naik(x_gpu, 10000, 30000), 'r', label='Tinggi')
            grafik[0,1].set_title('Kriteria: GPU 3DMark (10000 - 30000)')
            grafik[0,1].legend(); grafik[0,1].grid(True, linestyle='--', alpha=0.6)

            grafik[1,0].plot(x_ram, plot_turun(x_ram, 8, 64), 'b', label='Kecil')
            grafik[1,0].plot(x_ram, plot_naik(x_ram, 8, 64), 'r', label='Besar')
            grafik[1,0].set_title('Kriteria: Kapasitas RAM (8GB - 64GB)')
            grafik[1,0].legend(); grafik[1,0].grid(True, linestyle='--', alpha=0.6)

            grafik[1,1].plot(x_storage, plot_turun(x_storage, 256, 2048), 'b', label='Kecil')
            grafik[1,1].plot(x_storage, plot_naik(x_storage, 256, 2048), 'r', label='Besar')
            grafik[1,1].set_title('Kriteria: Kapasitas Storage (256GB - 2048GB)')
            grafik[1,1].legend(); grafik[1,1].grid(True, linestyle='--', alpha=0.6)

            grafik[2,0].plot(x_harga, plot_turun(x_harga, 200, 3000), 'b', label='Murah')
            grafik[2,0].plot(x_harga, plot_naik(x_harga, 200, 3000), 'r', label='Mahal')
            grafik[2,0].set_title('Kriteria: Harga USD (200 - 3000)')
            grafik[2,0].legend(); grafik[2,0].grid(True, linestyle='--', alpha=0.6)

            grafik[2,1].plot(x_out, plot_turun(x_out, 0, 100), 'b', label='Kelayakan Rendah')
            grafik[2,1].plot(x_out, plot_naik(x_out, 0, 100), 'r', label='Kelayakan Tinggi')
            grafik[2,1].set_title('Output Konsekuen: Skor Kelayakan')
            grafik[2,1].legend(); grafik[2,1].grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()
            st.pyplot(fig)

        # Proses Kalkulasi Skor Tsukamoto
        df_datafinal['Skor_Kelayakan'] = df_datafinal.apply(
            lambda row: hitung_tsukamoto(row['multiScore'], row['3DMark'], row['RAM_GB'], row['Storage_GB'], row['Price_USD']), axis=1
        )
        
        df_ranking = df_datafinal.sort_values(by="Skor_Kelayakan", ascending=False).reset_index(drop=True)
        
        # Batasan Kelayakan: Z >= 40 = Layak, Z < 40 = Tidak Layak
        BATAS_KELAYAKAN = 40
        df_ranking["Status"] = df_ranking["Skor_Kelayakan"].apply(
            lambda z: "✅ Layak" if z >= BATAS_KELAYAKAN else "❌ Tidak Layak"
        )
        
        df_ranking.index = df_ranking.index + 1
        st.session_state["df_ranking"] = df_ranking

        st.markdown("### 🏆 Hasil Akhir Rekomendasi Laptop Terbaik (Fuzzy Tsukamoto)")
        
        # Ringkasan statistik kelayakan
        jumlah_layak = (df_ranking["Skor_Kelayakan"] >= BATAS_KELAYAKAN).sum()
        jumlah_tidak_layak = (df_ranking["Skor_Kelayakan"] < BATAS_KELAYAKAN).sum()
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Laptop", len(df_ranking))
        with col_stat2:
            st.metric("✅ Layak (Z ≥ 40)", jumlah_layak)
        with col_stat3:
            st.metric("❌ Tidak Layak (Z < 40)", jumlah_tidak_layak)
        
        if not df_ranking.empty:
            df_layak = df_ranking[df_ranking["Skor_Kelayakan"] >= BATAS_KELAYAKAN]
            if not df_layak.empty:
                laptop_juara = df_layak.iloc[0]
                st.info(
                    f"🎯 Laptop paling direkomendasikan adalah **{laptop_juara['Model']}** "
                    f"dengan Skor Kelayakan **{laptop_juara['Skor_Kelayakan']:.2f}/100** — ✅ Layak."
                )
            else:
                st.warning("⚠️ Tidak ada laptop yang memenuhi batas kelayakan (Z ≥ 40).")
            
            st.dataframe(
                df_ranking[["Model", "multiScore", "3DMark", "RAM_GB", "Storage_GB", "Price_USD", "Skor_Kelayakan", "Status"]],
                width='stretch'
            )

        # ---------------------------------------------------------
        # DETAIL KALKULASI STEP-BY-STEP
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("## 🔍 Langkah - Langkah Pengerjaan")
        pilihan_laptop = st.selectbox("Pilih Laptop untuk membedah perhitungan Fuzzy Tsukamoto:", df_ranking["Model"].tolist())
        
        data_lp = df_ranking[df_ranking["Model"] == pilihan_laptop].iloc[0]
        
        # Logika pembersihan .0 di belakang koma untuk tampilan UI yang lebih rapi
        cpu_val = int(data_lp['multiScore']) if data_lp['multiScore'] % 1 == 0 else data_lp['multiScore']
        gpu_val = int(data_lp['3DMark']) if data_lp['3DMark'] % 1 == 0 else data_lp['3DMark']
        ram_val = int(data_lp['RAM_GB']) if data_lp['RAM_GB'] % 1 == 0 else data_lp['RAM_GB']
        storage_val = int(data_lp['Storage_GB']) if data_lp['Storage_GB'] % 1 == 0 else data_lp['Storage_GB']
        harga_val = int(data_lp['Price_USD']) if data_lp['Price_USD'] % 1 == 0 else data_lp['Price_USD']
        skor_akhir = data_lp['Skor_Kelayakan']
        
        # --- PERHITUNGAN ---
        cpu_r = kurva_turun(cpu_val, 8000, 30000); cpu_t = kurva_naik(cpu_val, 8000, 30000)
        gpu_r = kurva_turun(gpu_val, 10000, 30000); gpu_t = kurva_naik(gpu_val, 10000, 30000)
        ram_k = kurva_turun(ram_val, 8, 64); ram_b = kurva_naik(ram_val, 8, 64)
        sto_k = kurva_turun(storage_val, 256, 2048); sto_b = kurva_naik(storage_val, 256, 2048)
        hrg_m = kurva_turun(harga_val, 200, 3000); hrg_mh = kurva_naik(harga_val, 200, 3000)
        
        st.markdown(f"#### 1️⃣ Fuzzifikasi (Nilai Keanggotaan) - **{pilihan_laptop}**")
        
        
        with st.container(border=True):
            st.markdown("##### 📝 Detail Perhitungan Derajat Keanggotaan ($\mu$):")
            
            st.markdown(f"**1. CPU (Nilai Crisp: {cpu_val})** | Batas: [8.000, 30.000]")
            st.latex(rf"\mu_{{CPU\_Rendah}} = \frac{{30000 - {cpu_val}}}{{30000 - 8000}} = {cpu_r:.3f}")
            st.latex(rf"\mu_{{CPU\_Tinggi}} = \frac{{{cpu_val} - 8000}}{{30000 - 8000}} = {cpu_t:.3f}")
            
            st.markdown(f"**2. GPU (Nilai Crisp: {gpu_val})** | Batas: [10.000, 30.000]")
            st.latex(rf"\mu_{{GPU\_Rendah}} = \frac{{30000 - {gpu_val}}}{{30000 - 10000}} = {gpu_r:.3f}")
            st.latex(rf"\mu_{{GPU\_Tinggi}} = \frac{{{gpu_val} - 10000}}{{30000 - 10000}} = {gpu_t:.3f}")
            
            st.markdown(f"**3. RAM (Nilai Crisp: {ram_val} GB)** | Batas: [8, 64]")
            st.latex(rf"\mu_{{RAM\_Kecil}} = \frac{{64 - {ram_val}}}{{64 - 8}} = {ram_k:.3f}")
            st.latex(rf"\mu_{{RAM\_Besar}} = \frac{{{ram_val} - 8}}{{64 - 8}} = {ram_b:.3f}")
            
            st.markdown(f"**4. Storage (Nilai Crisp: {storage_val} GB)** | Batas: [256, 2.048]")
            st.latex(rf"\mu_{{Storage\_Kecil}} = \frac{{2048 - {storage_val}}}{{2048 - 256}} = {sto_k:.3f}")
            st.latex(rf"\mu_{{Storage\_Besar}} = \frac{{{storage_val} - 256}}{{2048 - 256}} = {sto_b:.3f}")
            
            st.markdown(f"**5. Harga (Nilai Crisp: ${harga_val})** | Batas: [200, 3.000]")
            st.latex(rf"\mu_{{Harga\_Murah}} = \frac{{3000 - {harga_val}}}{{3000 - 200}} = {hrg_m:.3f}")
            st.latex(rf"\mu_{{Harga\_Mahal}} = \frac{{{harga_val} - 200}}{{3000 - 200}} = {hrg_mh:.3f}")
            
            st.markdown(f"**6. Skor Kelayakan (Nilai Crisp: z** | Batas: [0, 100]")
            st.latex(rf"Rendah = \frac{{100 - z_i}}{{100 - 0}} = \alpha_i")
            st.latex(rf"Tinggi = \frac{{z_i - 0}}{{100 - 0}}= \alpha_i")

        # Tabel Rangkuman Fuzzifikasi
        df_fuzzifikasi = pd.DataFrame({
            "Kriteria": ["CPU", "GPU", "RAM", "Storage", "Harga"],
            "Nilai Crisp": [cpu_val, gpu_val, ram_val, storage_val, harga_val],
            "μ (Rendah/Kecil/Murah)": [f"{cpu_r:.3f}", f"{gpu_r:.3f}", f"{ram_k:.3f}", f"{sto_k:.3f}", f"{hrg_m:.3f}"],
            "μ (Tinggi/Besar/Mahal)": [f"{cpu_t:.3f}", f"{gpu_t:.3f}", f"{ram_b:.3f}", f"{sto_b:.3f}", f"{hrg_mh:.3f}"]
        })
        st.table(df_fuzzifikasi)
        
        st.markdown("#### 2️⃣ Inferensi (Penerapan Rules & Alpha Predikat)")
        a1 = min(cpu_t, gpu_t, ram_b, sto_b, hrg_m); z1 = z_naik(a1, 0, 100)
        a2 = gpu_r; z2 = z_turun(a2, 0, 100)
        a3 = hrg_mh; z3 = z_turun(a3, 0, 100)
        a4 = min(cpu_r, gpu_t, hrg_m); z4 = z_naik(a4, 0, 75)
        a5 = ram_k; z5 = z_turun(a5, 0, 100)
        a6 = sto_k; z6 = z_turun(a6, 0, 100)
        
        df_inferensi = pd.DataFrame({
            "Aturan": ["R1: CPU Tinggi dan GPU Tinggi dan RAM Tinggi dan Storage Besar dan Harga Murah maka Kelayakan Tinggi", 
                       "R2: GPU Rendah maka Kelayakan Rendah", 
                       "R3: Harga Mahal maka Kelayakan Rendah", 
                       "R4: CPU Rendah dan GPU Tinggi dan Harga Murah maka Kelayakan Tinggi", 
                       "R5: RAM Kecil maka Kelayakan Rendah", 
                       "R6: Storage Kecil maka Kelayakan Rendah"],
            "α (Nilai Min)": [f"{a1:.3f}", f"{a2:.3f}", f"{a3:.3f}", f"{a4:.3f}", f"{a5:.3f}", f"{a6:.3f}"],
            "Nilai Z": [f"{z1:.3f}", f"{z2:.3f}", f"{z3:.3f}", f"{z4:.3f}", f"{z5:.3f}", f"{z6:.3f}"],
            "α × Z": [f"{a1*z1:.3f}", f"{a2*z2:.3f}", f"{a3*z3:.3f}", f"{a4*z4:.3f}", f"{a5*z5:.3f}", f"{a6*z6:.3f}"]
        })
        st.table(df_inferensi)
        
        # --- BLOK PENJABARAN MATEMATIS ALUR R1 - R6 ---
        with st.container(border=True):
            st.markdown("##### 📝 Detail Langkah Rumus dan Substitusi Angka:")
            
            st.markdown("**Aturan 1:**")
            st.latex(rf"\alpha_1 = \min(\mu_{{CPU\_Tinggi}}, \mu_{{GPU\_Tinggi}}, \mu_{{RAM\_Besar}}, \mu_{{Storage\_Besar}}, \mu_{{Harga\_Murah}})")
            st.latex(rf"\alpha_1 = \min({cpu_t:.3f}, {gpu_t:.3f}, {ram_b:.3f}, {sto_b:.3f}, {hrg_m:.3f}) = {a1:.3f}")
            st.latex(rf"Z_1 = 0 + (\alpha_1 \times (100 - 0)) = 0 + ({a1:.3f} \times 100) = {z1:.3f}")
            
            st.markdown("**Aturan 2:**")
            st.latex(rf"\alpha_2 = \mu_{{GPU\_Rendah}} = {gpu_r:.3f}")
            st.latex(rf"Z_2 = 100 - (\alpha_2 \times (100 - 0)) = 100 - ({a2:.3f} \times 100) = {z2:.3f}")
            
            st.markdown("**Aturan 3:**")
            st.latex(rf"\alpha_3 = \mu_{{Harga\_Mahal}} = {hrg_mh:.3f}")
            st.latex(rf"Z_3 = 100 - (\alpha_3 \times (100 - 0)) = 100 - ({a3:.3f} \times 100) = {z3:.3f}")
            
            st.markdown("**Aturan 4:**")
            st.latex(rf"\alpha_4 = \min(\mu_{{CPU\_Rendah}}, \mu_{{GPU\_Tinggi}}, \mu_{{Harga\_Murah}})")
            st.latex(rf"\alpha_4 = \min({cpu_r:.3f}, {gpu_t:.3f}, {hrg_m:.3f}) = {a4:.3f}")
            st.latex(rf"Z_4 = 0 + (\alpha_4 \times (75 - 0)) = 0 + ({a4:.3f} \times 75) = {z4:.3f}")
            
            st.markdown("**Aturan 5:**")
            st.latex(rf"\alpha_5 = \mu_{{RAM\_Kecil}} = {ram_k:.3f}")
            st.latex(rf"Z_5 = 100 - (\alpha_5 \times (100 - 0)) = 100 - ({a5:.3f} \times 100) = {z5:.3f}")
            
            st.markdown("**Aturan 6:**")
            st.latex(rf"\alpha_6 = \mu_{{Storage\_Kecil}} = {sto_k:.3f}")
            st.latex(rf"Z_6 = 100 - (\alpha_6 \times (100 - 0)) = 100 - ({a6:.3f} \times 100) = {z6:.3f}")
        
        st.markdown("#### 3️⃣ Defuzzifikasi (Weighted Average)")
        st.markdown("Tahap terakhir adalah metode *Weighted Average* untuk mengubah nilai fuzzy menjadi skor tegas (crisp).")
        st.latex(r"Z_{akhir} = \frac{\sum_{i=1}^{6} (\alpha_i \times Z_i)}{\sum_{i=1}^{6} \alpha_i}")
        
        total_alpha = a1 + a2 + a3 + a4 + a5 + a6
        total_alpha_z = (a1*z1) + (a2*z2) + (a3*z3) + (a4*z4) + (a5*z5) + (a6*z6)
        
        if total_alpha > 0:
            st.latex(fr"Skor = \frac{{{total_alpha_z:.3f}}}{{{total_alpha:.3f}}} = {skor_akhir:.2f}")
        else:
            st.latex(r"Skor = 0")

        # Kesimpulan Kelayakan
        st.markdown("#### 4️⃣ Kesimpulan Kelayakan")
        BATAS_KELAYAKAN = 40
        with st.container(border=True):
            st.markdown(f"**Batas Kelayakan:** Z ≥ {BATAS_KELAYAKAN} → Layak | Z < {BATAS_KELAYAKAN} → Tidak Layak")
            st.latex(rf"Z_{{akhir}} = {skor_akhir:.2f}")
            if skor_akhir >= BATAS_KELAYAKAN:
                st.success(f"✅ **{pilihan_laptop}** dinyatakan **LAYAK** dengan skor **{skor_akhir:.2f}** (≥ {BATAS_KELAYAKAN})")
            else:
                st.error(f"❌ **{pilihan_laptop}** dinyatakan **TIDAK LAYAK** dengan skor **{skor_akhir:.2f}** (< {BATAS_KELAYAKAN})")

    else:
        st.warning("Silakan buka halaman 'Data RAW', 'Data CPU', dan 'Data GPU' terlebih dahulu.")