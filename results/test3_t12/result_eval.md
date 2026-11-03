# LAPORAN PENGUJIAN OCR PADA CPU

Dokumen ini dibuat untuk menguji kemampuan Baidu Unlimited OCR 2 membaca tanpa GPU. Server menggunakan prosesor AMD Ryzen 9 9900X dengan 12 core of RAM 64 GB. Sistem operasi Ubuntu Server 24.04 LTS, Python 3.12.3.

1. Tujuan: memastikan model berjalan sepenuhnya di CPU.

2. Metode: satu halaman gambar diproses dengan model.infer().

3. Metrik: waktu, RAM, CPU, dan kemiripan teks hasil ekstraksi.

Tabel contoh:

<table><tr><td>Nomor</td><td>Komponen</td><td>Nilai</td></tr><tr><td>1</td><td>CPU</td><td>Ryzen 9 9900X</td></tr><tr><td>2</td><td>RAM</td><td>64 GB DDR5 6000 MHz</td></tr><tr><td>3</td><td>Storage</td><td>Samsung 9100 PRO 2 TB</td></tr></table>

The quick brown fox jumps over the lazy dog. 0123456789

Alamat: Jl. Merdeka No. 17, Jakarta 10110. Telepon (021) 555-0199.

Total biaya: Rp 1.250.000,00 (satu juta dua ratus lima puluh ribu rupiah).