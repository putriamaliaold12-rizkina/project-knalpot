import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Segmentasi Produk — Berkah Performance",
    page_icon="🏍️",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    kmeans = joblib.load('model/kmeans_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    return kmeans, scaler

kmeans, scaler = load_model()

# ============================================================
# FUNGSI KATEGORISASI
# ============================================================
def kategorisasi(nama):
    nama_bersih = str(nama).strip().lower()
    nama_bersih = ' '.join(nama_bersih.split())

    for prefix in ['knalpot mobil ', 'knalpot ', 'knalpotn ',
                   'knalpo ', '(po) ', '(po)', 'po ', 'po. ',
                   'custom.']:
        if nama_bersih.startswith(prefix):
            nama_bersih = nama_bersih[len(prefix):]
            break

    if any(nama_bersih.startswith(x) for x in [
        'fullsystem', 'full system', 'fullsistem',
        'fulsystem', 'fullsystemm', 'ullsystem',
        'fullset', 'full set', 'fulset', 'full piu piu'
    ]):
        return 'Fullsystem'
    elif any(nama_bersih.startswith(x) for x in [
        'tailpipe plus centerpipe', 'tailpipe + centerpipe',
        'tailpipe+centerpipe', 'tailpipe centerpipe',
        'tail + centerpipe', 'taillpipe+centerpipe',
        'tailpipe+center', 'tail pipe plus'
    ]):
        return 'Tailpipe + Centerpipe'
    elif nama_bersih.startswith('centerpipe'):
        return 'Centerpipe + Resonator'
    elif any(nama_bersih.startswith(x) for x in [
        'downpipe frontpipe', 'downpipe + frontpipe',
        'downpipe+frontpipe', 'dp fp',
        'downpipe fronpipe', 'downpipe plus frontpipe',
        'dwonpipe dan frontpipe'
    ]):
        return 'Downpipe + Frontpipe'
    elif any(nama_bersih.startswith(x) for x in [
        'downpipe', 'dwonpipe', 'ownpipe'
    ]):
        return 'Downpipe Satuan'
    elif nama_bersih.startswith('frontpipe'):
        return 'Frontpipe Satuan'
    elif any(nama_bersih.startswith(x) for x in [
        'tailpipe', 'tail pipe', 'taipipe',
        'taillpipe', 'ailpipe'
    ]):
        return 'Tailpipe Satuan'
    elif any(nama_bersih.startswith(x) for x in [
        'side exit', 'sside exit'
    ]):
        return 'Side Exit'
    elif any(nama_bersih.startswith(x) for x in [
        'bolt on', 'bolton', 'no 1 bolton',
        'no 1. pipa bolt', 'no 1 bolt',
        'standar bolt on', 'standar brio', 'standar '
    ]):
        return 'Bolt On'
    elif any(nama_bersih.startswith(x) for x in [
        'header', 'plendes'
    ]):
        return 'Header'
    elif any(nama_bersih.startswith(x) for x in [
        'resonator', 'resonantor', 'reson pajero'
    ]):
        return 'Resonator Satuan'
    elif nama_bersih.startswith('muffler'):
        return 'Muffler Satuan'
    elif any(nama_bersih.startswith(x) for x in [
        'pipa bolt', 'pipa dan bolt', 'pipa sambungan',
        'pipa bolton', 'pipa tanpa muffler',
        'piping intercooler', 'pipa '
    ]):
        return 'Pipa Bolt On'
    elif any(nama_bersih.startswith(x) for x in [
        'tabung', 'tabung knalpot',
        'tabung standar', 'tabung standart'
    ]):
        return 'Tabung Knalpot'
    elif nama_bersih.startswith('db killer'):
        return 'Db Killer'
    elif any(x in nama_bersih for x in [
        'hks legamax', 'hks hi power',
        'hks gronel', 'fmf f4', 'muffler hks', 'hks '
    ]):
        return 'Muffler Satuan'
    elif 'fleksibel pnp' in nama_bersih:
        return 'Lainnya'
    elif 'isuzu traga' in nama_bersih:
        return 'Lainnya'
    elif 'cumi cumi' in nama_bersih:
        return 'Lainnya'
    else:
        return 'Lainnya'

# ============================================================
# FUNGSI PREDIKSI CLUSTER
# ============================================================
def prediksi_cluster(df):
    df = df.copy()
    df['KATEGORI'] = df['NAMA BARANG'].apply(kategorisasi)

    df_agg = df.groupby('KATEGORI').agg(
        total_transaksi=('HARGA', 'count'),
        total_pendapatan=('HARGA', 'sum')
    ).reset_index()

    X = df_agg[['total_transaksi', 'total_pendapatan']]
    X_scaled = scaler.transform(X)

    df_agg['Cluster'] = kmeans.predict(X_scaled)

    cluster_map = {1: 'Terlaris', 0: 'Sedang', 2: 'Kurang Laris'}
    df_agg['Label'] = df_agg['Cluster'].map(cluster_map)
    if df_agg['Label'].isna().any():
        df_agg['Label'] = df_agg['Label'].fillna('Kurang Laris')

    return df_agg

# ============================================================
# UI STREAMLIT
# ============================================================
st.title("🏍️ Segmentasi Produk Berkah Performance")
st.caption("K-Means Clustering — CRISP-DM | Putri Amelia | 2025")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload file Excel (.xlsx)",
        type=['xlsx'],
        help="Upload laporan penjualan bulanan atau tahunan"
    )
    st.markdown("---")
    st.markdown("**Kolom yang dibutuhkan:**")
    st.markdown("- `NAMA BARANG`")
    st.markdown("- `HARGA`")
    st.markdown("- `BULAN` (opsional)")
    st.markdown("---")
    st.markdown("**Info Model:**")
    st.markdown("- Algoritma : K-Means (K=3)")
    st.markdown("- Dilatih   : Data 2025")
    st.markdown("- Kategori  : 16 kategori produk")

# ============================================================
# HALAMAN UTAMA
# ============================================================
if uploaded_file is None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📤 **Step 1**\nUpload file Excel di sidebar kiri")
    with col2:
        st.info("⚙️ **Step 2**\nSistem otomatis proses dan prediksi cluster")
    with col3:
        st.info("📊 **Step 3**\nLihat hasil segmentasi produk")

    st.markdown("---")
    st.subheader("ℹ️ Tentang Aplikasi")
    st.write(
        "Aplikasi ini menggunakan algoritma **K-Means Clustering** untuk "
        "mengelompokkan produk penjualan Toko Berkah Performance ke dalam "
        "3 cluster: **Terlaris**, **Sedang**, dan **Kurang Laris**."
    )
    st.markdown("---")
    st.subheader("📦 16 Kategori Produk")
    kategori_list = [
        "1. Fullsystem", "2. Tailpipe + Centerpipe",
        "3. Centerpipe + Resonator", "4. Downpipe + Frontpipe",
        "5. Downpipe Satuan", "6. Frontpipe Satuan",
        "7. Tailpipe Satuan", "8. Side Exit",
        "9. Bolt On", "10. Header",
        "11. Resonator Satuan", "12. Muffler Satuan",
        "13. Pipa Bolt On", "14. Tabung Knalpot",
        "15. Db Killer", "16. Lainnya"
    ]
    col1, col2 = st.columns(2)
    for i, k in enumerate(kategori_list):
        if i < 8:
            col1.markdown(f"- {k}")
        else:
            col2.markdown(f"- {k}")

else:
    # Proses data
    with st.spinner("⚙️ Memproses data..."):
        df = pd.read_excel(uploaded_file)

        if 'NAMA BARANG' not in df.columns or 'HARGA' not in df.columns:
            st.error("❌ File harus memiliki kolom NAMA BARANG dan HARGA!")
            st.stop()

        df_hasil = prediksi_cluster(df)

    st.success(f"✅ Data berhasil diproses! {len(df):,} transaksi, {len(df_hasil)} kategori produk.")
    st.markdown("---")

    # Filter bulan
    if 'BULAN' in df.columns:
        bulan_list = ['Semua Bulan'] + sorted(df['BULAN'].unique().tolist())
        bulan_pilih = st.selectbox("🗓️ Filter Bulan", bulan_list)
        if bulan_pilih != 'Semua Bulan':
            df_filter = df[df['BULAN'] == bulan_pilih]
            df_hasil = prediksi_cluster(df_filter)
            st.info(f"Menampilkan data bulan: **{bulan_pilih}**")

    # Metrik
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transaksi", f"{len(df):,}")
    with col2:
        st.metric("Total Kategori", f"{len(df_hasil)}")
    with col3:
        total_pendapatan = df_hasil['total_pendapatan'].sum()
        st.metric("Total Pendapatan", f"Rp {total_pendapatan/1_000_000:.1f}jt")
    with col4:
        st.metric("Jumlah Cluster", "K = 3")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Hasil Clustering", "📈 Visualisasi", "📋 Data Lengkap"])

    with tab1:
        for label, color, emoji in [
            ('Terlaris',     '#2ECC71', '🟢'),
            ('Sedang',       '#F39C12', '🟡'),
            ('Kurang Laris', '#E74C3C', '🔴')
        ]:
            subset = df_hasil[df_hasil['Label'] == label].sort_values(
                'total_transaksi', ascending=False
            )
            st.subheader(f"{emoji} {label} ({len(subset)} kategori)")
            if len(subset) > 0:
                subset_display = subset[['KATEGORI', 'total_transaksi', 'total_pendapatan']].copy()
                subset_display.columns = ['Kategori Produk', 'Total Transaksi', 'Total Pendapatan (Rp)']
                subset_display['Total Pendapatan (Rp)'] = subset_display['Total Pendapatan (Rp)'].apply(
                    lambda x: f"Rp {x:,.0f}".replace(',', '.')
                )
                st.dataframe(subset_display, use_container_width=True, hide_index=True)
            else:
                st.info("Tidak ada kategori dalam cluster ini.")

    with tab2:
        warna = {
            'Terlaris'    : '#2ECC71',
            'Sedang'      : '#F39C12',
            'Kurang Laris': '#E74C3C'
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            'Hasil K-Means Clustering — Segmentasi Produk\nBerkah Performance',
            fontsize=13, fontweight='bold'
        )

        for label, color in warna.items():
            subset = df_hasil[df_hasil['Label'] == label]
            for ax in [ax1, ax2]:
                ax.scatter(
                    subset['total_transaksi'],
                    subset['total_pendapatan'] / 1_000_000,
                    c=color, label=label, s=120, alpha=0.85,
                    edgecolors='white', linewidth=1.5
                )

        for _, row in df_hasil.iterrows():
            ax1.annotate(
                row['KATEGORI'],
                (row['total_transaksi'], row['total_pendapatan'] / 1_000_000),
                textcoords="offset points", xytext=(5, 4), fontsize=7
            )

        df_zoom = df_hasil[df_hasil['KATEGORI'] != 'Fullsystem']
        for _, row in df_zoom.iterrows():
            ax2.annotate(
                row['KATEGORI'],
                (row['total_transaksi'], row['total_pendapatan'] / 1_000_000),
                textcoords="offset points", xytext=(5, 4), fontsize=7.5
            )

        for ax in [ax1, ax2]:
            ax.set_xlabel('Total Transaksi', fontsize=10)
            ax.set_ylabel('Total Pendapatan (Juta Rp)', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'Rp {x:.0f}jt')
            )

        ax1.set_title('Keseluruhan Data', fontsize=11, fontweight='bold')
        ax2.set_title('Zoom — Sedang & Kurang Laris', fontsize=11, fontweight='bold')

        legend_patches = [mpatches.Patch(color=c, label=l) for l, c in warna.items()]
        fig.legend(
            handles=legend_patches, loc='lower center',
            ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.05)
        )

        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.dataframe(df_hasil, use_container_width=True, hide_index=True)
        st.download_button(
            label="⬇️ Download Hasil (.csv)",
            data=df_hasil.to_csv(index=False).encode('utf-8'),
            file_name='hasil_clustering.csv',
            mime='text/csv'
        )

    st.markdown("---")

    # CEK KATEGORI PRODUK
    st.subheader("🔍 Cek Kategori & Cluster Produk")
    st.caption("Ketik nama produk untuk mengetahui kategori dan cluster-nya berdasarkan data yang diupload")

    col1, col2 = st.columns([3, 1])
    with col1:
        input_produk = st.text_input(
            "Nama Produk",
            placeholder="Contoh: Fullsystem Knalpot Brio, Tailpipe Innova 2Kd, dll"
        )
    with col2:
        cek_button = st.button("🔍 Cek Sekarang", use_container_width=True)

    if cek_button and input_produk:
        kategori_hasil = kategorisasi(input_produk)
        info = df_hasil[df_hasil['KATEGORI'] == kategori_hasil]

        if len(info) > 0:
            row = info.iloc[0]
            label = row['Label']
            config = {
                'Terlaris'    : ('🟢', 'success'),
                'Sedang'      : ('🟡', 'warning'),
                'Kurang Laris': ('🔴', 'error')
            }
            emoji, tipe = config.get(label, ('⚪', 'info'))

            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.info(f"**Kategori**\n\n{kategori_hasil}")
            with col2:
                if tipe == 'success':
                    st.success(f"**Cluster**\n\n{emoji} {label}")
                elif tipe == 'warning':
                    st.warning(f"**Cluster**\n\n{emoji} {label}")
                else:
                    st.error(f"**Cluster**\n\n{emoji} {label}")
            with col3:
                st.info(f"**Total Transaksi**\n\n{int(row['total_transaksi']):,} transaksi")
            with col4:
                pendapatan_fmt = f"Rp {int(row['total_pendapatan']):,}".replace(',', '.')
                st.info(f"**Total Pendapatan**\n\n{pendapatan_fmt}")

            st.caption(
                f"⚠️ Catatan: Produk **'{input_produk}'** diidentifikasi masuk kategori "
                f"**{kategori_hasil}**. Data transaksi dan pendapatan yang ditampilkan "
                f"mencakup **seluruh produk dalam kategori {kategori_hasil}**, "
                f"bukan hanya produk yang Anda cari."
            )
        else:
            st.warning(
                f"Produk masuk kategori **{kategori_hasil}** namun "
                f"kategori ini tidak ditemukan dalam data yang diupload."
            )

    elif cek_button and not input_produk:
        st.warning("⚠️ Silakan ketik nama produk terlebih dahulu!")
