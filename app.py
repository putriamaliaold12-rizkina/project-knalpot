import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import math

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
# FUNGSI PAGINATION
# ============================================================
def tampilkan_pagination(df_filter, key_prefix, per_page=10):
    total = len(df_filter)
    total_page = max(1, math.ceil(total / per_page))

    if total == 0:
        st.info("Tidak ada data untuk kategori ini.")
        return

    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        page = st.number_input(
            f"Halaman (total {total_page} halaman, {total} data)",
            min_value=1,
            max_value=total_page,
            value=1,
            step=1,
            key=f"{key_prefix}_page"
        )

    start = (page - 1) * per_page
    end = start + per_page
    df_page = df_filter.iloc[start:end].reset_index(drop=True)
    df_page.index = df_page.index + start + 1

    st.dataframe(df_page, use_container_width=True)
    st.caption(
        f"Menampilkan data {start+1}–{min(end, total)} dari {total} transaksi"
    )

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
        help="Upload laporan penjualan tahunan (12 bulan penuh)"
    )
    st.markdown("---")
    st.markdown("**Kolom yang dibutuhkan:**")
    st.markdown("- `NAMA BARANG`")
    st.markdown("- `HARGA`")
    st.markdown("---")
    st.caption(
        "⚠️ Model dilatih dari data 2025. "
        "Untuk hasil optimal, upload data "
        "12 bulan penuh."
    )

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
        st.info("📊 **Step 3**\nLihat hasil segmentasi & eksplorasi produk")

    st.markdown("---")
    st.subheader("ℹ️ Tentang Aplikasi")
    st.write(
        "Aplikasi ini menggunakan algoritma **K-Means Clustering** untuk "
        "mengelompokkan produk penjualan Toko Berkah Performance ke dalam "
        "3 cluster: **Terlaris**, **Sedang**, dan **Kurang Laris** "
        "berdasarkan total transaksi dan total pendapatan."
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

        df['KATEGORI'] = df['NAMA BARANG'].apply(kategorisasi)
        df_hasil = prediksi_cluster(df)

        # Gabungkan label ke data transaksi
        df = df.merge(
            df_hasil[['KATEGORI', 'Label']],
            on='KATEGORI', how='left'
        )

    st.success(
        f"✅ Data berhasil diproses! "
        f"{len(df):,} transaksi | "
        f"{len(df_hasil)} kategori produk"
    )
    st.markdown("---")

    # Metrik
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transaksi", f"{len(df):,}")
    with col2:
        st.metric("Total Kategori", f"{len(df_hasil)}")
    with col3:
        total_pendapatan = df_hasil['total_pendapatan'].sum()
        st.metric("Total Pendapatan", f"Rp {total_pendapatan/1_000_000:.1f}jt")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Hasil Clustering",
        "📈 Visualisasi",
        "🔍 Eksplorasi Produk",
        "📋 Ringkasan Kategori"
    ])

    # ── TAB 1: Hasil Clustering ──
    with tab1:
        for label, emoji in [
            ('Terlaris',     '🟢'),
            ('Sedang',       '🟡'),
            ('Kurang Laris', '🔴')
        ]:
            subset = df_hasil[df_hasil['Label'] == label].sort_values(
                'total_transaksi', ascending=False
            )
            st.subheader(f"{emoji} {label} ({len(subset)} kategori)")
            if len(subset) > 0:
                subset_display = subset[[
                    'KATEGORI', 'total_transaksi', 'total_pendapatan'
                ]].copy()
                subset_display.columns = [
                    'Kategori Produk', 'Total Transaksi', 'Total Pendapatan (Rp)'
                ]
                subset_display['Total Pendapatan (Rp)'] = (
                    subset_display['Total Pendapatan (Rp)']
                    .apply(lambda x: f"Rp {x:,.0f}".replace(',', '.'))
                )
                st.dataframe(
                    subset_display,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Tidak ada kategori dalam cluster ini.")

    # ── TAB 2: Visualisasi ──
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
        ax2.set_title(
            'Zoom — Sedang & Kurang Laris',
            fontsize=11, fontweight='bold'
        )

        legend_patches = [
            mpatches.Patch(color=c, label=l) for l, c in warna.items()
        ]
        fig.legend(
            handles=legend_patches, loc='lower center',
            ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.05)
        )

        plt.tight_layout()
        st.pyplot(fig)

    # ── TAB 3: Eksplorasi Produk ──
    with tab3:
        st.subheader("🔍 Eksplorasi Produk per Kategori")
        st.caption(
            "Pilih kategori untuk melihat semua transaksi produk "
            "dalam kategori tersebut beserta cluster-nya."
        )

        # Dropdown kategori
        col1, col2 = st.columns([2, 1])
        with col1:
            kategori_pilih = st.selectbox(
                "Pilih Kategori Produk",
                options=sorted(df['KATEGORI'].unique().tolist()),
                key="kategori_select"
            )
        with col2:
            # Info cluster kategori ini
            info_cluster = df_hasil[df_hasil['KATEGORI'] == kategori_pilih]
            if len(info_cluster) > 0:
                label_kat = info_cluster.iloc[0]['Label']
                emoji_map = {
                    'Terlaris': '🟢',
                    'Sedang': '🟡',
                    'Kurang Laris': '🔴'
                }
                emoji_kat = emoji_map.get(label_kat, '⚪')
                st.metric("Cluster", f"{emoji_kat} {label_kat}")

        st.markdown("---")

        # Filter data per kategori
        df_kategori = df[df['KATEGORI'] == kategori_pilih].drop(
            columns=['KATEGORI', 'Label'], errors='ignore'
        ).reset_index(drop=True)

        # Info ringkasan
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transaksi", f"{len(df_kategori):,}")
        with col2:
            total_pend_kat = df_kategori['HARGA'].sum()
            st.metric("Total Pendapatan", f"Rp {total_pend_kat/1_000_000:.1f}jt")
        with col3:
            rata_harga = df_kategori['HARGA'].mean()
            st.metric("Rata-rata Harga", f"Rp {rata_harga:,.0f}".replace(',', '.'))

        st.markdown("---")

        # Pagination
        tampilkan_pagination(df_kategori, key_prefix=kategori_pilih)

        # Download per kategori
        st.download_button(
            label=f"⬇️ Download Data {kategori_pilih} (.csv)",
            data=df_kategori.to_csv(index=False).encode('utf-8'),
            file_name=f"data_{kategori_pilih.replace(' ', '_')}.csv",
            mime='text/csv'
        )

    # ── TAB 4: Ringkasan Kategori ──
    with tab4:
        st.subheader("📋 Ringkasan Semua Kategori")

        df_ringkasan = df_hasil[[
            'KATEGORI', 'total_transaksi', 'total_pendapatan', 'Label'
        ]].copy()
        df_ringkasan.columns = [
            'Kategori', 'Total Transaksi',
            'Total Pendapatan (Rp)', 'Cluster'
        ]
        df_ringkasan['Total Pendapatan (Rp)'] = (
            df_ringkasan['Total Pendapatan (Rp)']
            .apply(lambda x: f"Rp {x:,.0f}".replace(',', '.'))
        )
        df_ringkasan = df_ringkasan.sort_values(
            'Total Transaksi', ascending=False
        ).reset_index(drop=True)
        df_ringkasan.index += 1

        st.dataframe(df_ringkasan, use_container_width=True)

        st.download_button(
            label="⬇️ Download Ringkasan (.csv)",
            data=df_ringkasan.to_csv(index=False).encode('utf-8'),
            file_name='ringkasan_clustering.csv',
            mime='text/csv'
        )
