<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <meta name="theme-color" content="#0284C7" />
  <title>RespiKidd Offline</title>
  <link rel="manifest" href="manifest.json" />
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&display=swap');
    * { box-sizing: border-box; font-family: 'Nunito', -apple-system, BlinkMacSystemFont, sans-serif; -webkit-tap-highlight-color: transparent; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #CFFAFE 0%, #E0F2FE 55%, #F0FDF4 100%);
      min-height: 100vh;
      color: #1E293B;
      padding: 16px 16px 36px;
    }
    .container { max-width: 460px; margin: 0 auto; }

    /* Top Bar & Progress Dots */
    .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .badge {
      background: #DCFCE7; color: #15803D; font-weight: 800; font-size: 11px;
      padding: 6px 12px; border-radius: 999px; border: 1.5px solid #86EFAC;
      display: inline-flex; align-items: center; gap: 6px;
    }
    .dot-live { width: 8px; height: 8px; background: #22C55E; border-radius: 50%; animation: pulse 1.5s infinite; }
    .steps-indicator { display: flex; gap: 6px; }
    .step-pill { width: 22px; height: 8px; border-radius: 99px; background: #BAE6FD; transition: all 0.3s ease; }
    .step-pill.active { width: 34px; background: #0284C7; }

    /* Mascot & Header */
    .hero-mascot {
      width: 115px; height: 115px; object-fit: contain; border-radius: 28px;
      background: #FFFFFF; padding: 8px; box-shadow: 0 10px 25px rgba(2, 132, 199, 0.18);
      border: 3px solid #BAE6FD; animation: floatMascot 3s ease-in-out infinite;
    }
    .header { text-align: center; margin-bottom: 16px; }
    .header h1 { color: #0369A1; margin: 8px 0 2px; font-size: 28px; font-weight: 900; letter-spacing: -0.5px; }
    .header .subtitle { font-size: 13px; color: #0284C7; font-weight: 700; margin: 0; }

    /* Cute Tactile Cards */
    .card {
      background: #FFFFFF; border-radius: 26px; padding: 20px; margin-bottom: 14px;
      box-shadow: 0 8px 20px rgba(14, 116, 144, 0.08); border: 2px solid #E0F2FE;
      animation: fadeSlide 0.35s ease;
    }
    .card.yellow { background: #FEFCE8; border-color: #FEF08A; }
    .card.blue { background: #F0F9FF; border-color: #BAE6FD; }
    .card.green { background: #ECFDF5; border-color: #A7F3D0; }
    .card.red { background: #FFF1F2; border-color: #FECDD3; }

    h2 { margin: 0 0 8px; font-size: 19px; font-weight: 900; color: #0369A1; }
    p { margin: 0 0 10px; font-size: 14px; line-height: 1.55; color: #475569; font-weight: 600; }

    /* Form Inputs */
    label { display: block; font-size: 13px; font-weight: 800; margin: 12px 0 5px; color: #0F172A; }
    input[type="text"], input[type="number"], select {
      width: 100%; padding: 13px 14px; border: 2px solid #CBD5E1; border-radius: 16px;
      font-size: 15px; font-weight: 700; background: #F8FAFC; color: #1E293B; transition: all 0.2s;
    }
    input:focus, select:focus { outline: none; border-color: #0284C7; background: #FFFFFF; box-shadow: 0 0 0 4px rgba(2,132,199,0.12); }

    .check-card {
      display: flex; align-items: center; gap: 12px; margin-top: 10px; padding: 12px 14px;
      background: #F8FAFC; border: 2px solid #E2E8F0; border-radius: 16px; cursor: pointer; font-size: 14px; font-weight: 700;
    }
    .check-card input { width: 20px; height: 20px; accent-color: #0284C7; }

    /* 3D Tactile Cute Buttons */
    .btn {
      width: 100%; padding: 15px; border: none; border-radius: 18px; font-size: 16px; font-weight: 800;
      color: #FFF; cursor: pointer; margin-top: 12px; transition: transform 0.12s, box-shadow 0.12s;
      display: flex; align-items: center; justify-content: center; gap: 8px;
    }
    .btn:active { transform: translateY(3px); box-shadow: none !important; }
    .btn-primary { background: #0284C7; box-shadow: 0 5px 0 #0369A1; }
    .btn-success { background: #10B981; box-shadow: 0 5px 0 #047857; }
    .btn-warn { background: #F59E0B; box-shadow: 0 5px 0 #B45309; }
    .btn-danger { background: #E11D48; box-shadow: 0 5px 0 #9F1239; animation: pulseBtn 1.2s infinite; }
    .btn-indigo { background: #6366F1; box-shadow: 0 5px 0 #4338CA; }
    .btn-gray { background: #94A3B8; box-shadow: 0 4px 0 #64748B; font-size: 14px; padding: 12px; }

    /* Audio Visualizer Bars */
    .wave-box {
      display: none; justify-content: center; align-items: center; gap: 5px; height: 38px;
      margin-top: 12px; background: #FFF1F2; border-radius: 14px; border: 1.5px dashed #FDA4AF;
    }
    .wave-box.recording { display: flex; }
    .wave-bar { width: 5px; height: 10px; background: #E11D48; border-radius: 99px; animation: waveAnim 0.6s ease-in-out infinite alternate; }
    .wave-bar:nth-child(2) { animation-delay: 0.15s; }
    .wave-bar:nth-child(3) { animation-delay: 0.3s; }
    .wave-bar:nth-child(4) { animation-delay: 0.1s; }
    .wave-bar:nth-child(5) { animation-delay: 0.25s; }

    .status-box {
      padding: 12px 14px; border-radius: 16px; background: #FEF3C7; border: 1.5px solid #FDE68A;
      color: #92400E; font-size: 13px; font-weight: 800; margin-top: 12px; text-align: center;
    }

    /* Guide Step Icon Badge */
    .guide-row { display: flex; gap: 12px; align-items: flex-start; }
    .num-circle {
      min-width: 34px; height: 34px; border-radius: 12px; background: #0284C7; color: #FFF;
      font-weight: 900; font-size: 16px; display: flex; align-items: center; justify-content: center;
    }

    .step { display: none; }
    .step.active { display: block; }
    .faskes-item { border-bottom: 1.5px dashed #E2E8F0; padding: 10px 0; font-size: 13px; }
    .faskes-item:last-child { border-bottom: none; }
    .call-link {
      display: inline-block; margin-top: 6px; padding: 6px 12px; background: #E0F2FE;
      color: #0369A1; border-radius: 10px; font-weight: 800; text-decoration: none; font-size: 12px;
    }

    @keyframes floatMascot { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-6px); } }
    @keyframes fadeSlide { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes waveAnim { from { height: 8px; } to { height: 26px; } }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    @keyframes pulseBtn { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.02); } }
  </style>
</head>
<body>
<div class="container">
  <div class="topbar">
    <span class="badge"><span class="dot-live"></span> 100% OFFLINE</span>
    <div class="steps-indicator">
      <div id="pill1" class="step-pill active"></div>
      <div id="pill2" class="step-pill"></div>
      <div id="pill3" class="step-pill"></div>
      <div id="pill4" class="step-pill"></div>
    </div>
  </div>

  <div class="header">
    <img src="icon-512.png" alt="Maskot RespiKidd" class="hero-mascot" onerror="this.style.display='none'" />
    <h1>RespiKidd</h1>
    <p class="subtitle">Sahabat Skrining Pernapasan Anak</p>
  </div>

  <!-- STEP 1: WELCOME -->
  <div id="step1" class="step active">
    <div class="card yellow" style="text-align:center;">
      <h2>Halo Ayah & Bunda! 👋</h2>
      <p>RespiKidd siap membantu skrining dini gejala bronkitis pada batita melalui suara batuk <b>tanpa kuota internet</b>, lengkap dengan kontak Puskesmas & RS rujukan seluruh Indonesia.</p>
      <button class="btn btn-primary" onclick="navStep(2)">Mulai Pemeriksaan &raquo;</button>
    </div>
  </div>

  <!-- STEP 2: GUIDE -->
  <div id="step2" class="step">
    <div class="card blue">
      <div class="guide-row">
        <div class="num-circle">1</div>
        <div>
          <h2>Isi Data Anak & Wilayah</h2>
          <p style="margin:0;">Membantu pencatatan rekam medis offline dan menyesuaikan kontak RS rujukan terdekat.</p>
        </div>
      </div>
    </div>
    <div class="card yellow">
      <div class="guide-row">
        <div class="num-circle" style="background:#D97706;">2</div>
        <div>
          <h2>Rekam Suara Batuk</h2>
          <p style="margin:0;">Dekatkan HP 15–20 cm saat anak batuk selama 2–3 detik di ruangan yang tenang.</p>
        </div>
      </div>
    </div>
    <div class="card green">
      <div class="guide-row">
        <div class="num-circle" style="background:#059669;">3</div>
        <div>
          <h2>Hasil AI & Kontak Darurat</h2>
          <p style="margin:0;">Lihat hasil analisis dalam hitungan detik, simpan laporan, atau hubungi 119/RSUD.</p>
        </div>
      </div>
    </div>
    <button class="btn btn-success" onclick="navStep(3)">Lanjut Isi Data Anak &raquo;</button>
    <button class="btn btn-gray" onclick="navStep(1)">&laquo; Kembali</button>
  </div>

  <!-- STEP 3: INPUT & AUDIO -->
  <div id="step3" class="step">
    <div class="card">
      <h2>Profil Anak & Suara Batuk</h2>
      <label>Nama Lengkap Anak *</label>
      <input type="text" id="childName" placeholder="Contoh: Budi Santoso" />

      <label>Nama Orang Tua / Wali *</label>
      <input type="text" id="parentName" placeholder="Contoh: Siti Aminah" />

      <label>Usia Anak (Bulan)</label>
      <input type="number" id="childAge" placeholder="Contoh: 18" />

      <label>Wilayah Rujukan</label>
      <select id="regionSelect"></select>

      <label>Desa / Kecamatan / Puskesmas</label>
      <input type="text" id="villageInput" placeholder="Contoh: Cimenyan" />

      <label class="check-card">
        <input type="checkbox" id="chkWheezing" />
        <span>Napas terdengar ngik-ngik / berlendir</span>
      </label>
      <label class="check-card">
        <input type="checkbox" id="chkFever" />
        <span>Demam atau batuk lebih dari 3 hari</span>
      </label>

      <div id="waveBox" class="wave-box">
        <div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div><div class="wave-bar"></div>
        <span style="font-size:12px; font-weight:800; color:#BE123C; margin-left:6px;">Mendengarkan suara...</span>
      </div>

      <div id="audioStatus" class="status-box">🎙️ Belum ada suara batuk direkam atau dipilih.</div>

      <button id="btnRecord" class="btn btn-warn" onclick="toggleRecording()">🎙️ Rekam Suara Batuk (Mic HP)</button>
      <input type="file" id="wavPicker" accept=".wav,audio/wav" style="display:none" onchange="handleFileUpload(event)" />
      <button class="btn btn-indigo" onclick="document.getElementById('wavPicker').click()">📂 Pilih File Audio (.wav)</button>
      <button class="btn btn-primary" onclick="runOfflineAnalysis()">🔍 Analisis Suara Offline Sekarang</button>
      <button class="btn btn-gray" onclick="navStep(2)">&laquo; Kembali</button>
    </div>
  </div>

  <!-- STEP 4: RESULT -->
  <div id="step4" class="step">
    <div id="resultCard" class="card">
      <h2 id="resTitle" style="text-align:center;"></h2>
      <p id="resScore" style="text-align:center; font-weight:900; font-size:17px; color:#0F172A;"></p>
      <p style="font-size:12px; text-align:center;">Catatan: Hasil ini merupakan skrining awal berbasis AI offline, bukan diagnosis medis final. Segera konsultasikan ke dokter/Puskesmas.</p>
    </div>

    <div class="card blue">
      <h2>📋 Ringkasan Data Pasien</h2>
      <p id="patientSummary" style="white-space: pre-line;"></p>
    </div>

    <div class="card yellow">
      <h2>🏥 Kontak Faskes & Darurat Offline</h2>
      <div id="faskesList"></div>
    </div>

    <button class="btn btn-success" onclick="downloadReport()">💾 Simpan Laporan Rekam Medis (.txt)</button>
    <button class="btn btn-primary" onclick="navStep(3)">🔄 Periksa Anak Lainnya</button>
  </div>
</div>

<script src="weights.js"></script>
<script src="app.js"></script>
<script>
  function navStep(n) {
    goStep(n);
    for (let i = 1; i <= 4; i++) {
      document.getElementById('pill' + i).classList.toggle('active', i <= n);
    }
  }
  const origToggle = toggleRecording;
  toggleRecording = async function() {
    await origToggle();
    document.getElementById('waveBox').classList.toggle('recording', isRecording);
  };
  const origRun = runOfflineAnalysis;
  runOfflineAnalysis = function() {
    origRun();
    if (document.getElementById('step4').classList.contains('active')) {
      for (let i = 1; i <= 4; i++) document.getElementById('pill' + i).classList.add('active');
    }
  };
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js');
  }
</script>
</body>
</html>
