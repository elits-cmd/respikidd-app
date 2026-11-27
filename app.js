const OFFLINE_FASKES_DB = {
  "Layanan Darurat Nasional (24 Jam)": [
    ["PSC 119 Kemenkes RI", "119", "Ambulans & Gawat Darurat Nasional"],
    ["Panggilan Darurat Terpadu", "112", "Aktif Tanpa Kuota / Sinyal Darurat"],
    ["Halo Kemenkes RI", "1500567", "Pusat Informasi Kesehatan RI"],
    ["RSAB Harapan Kita Jakarta", "0215668284", "Rujukan Nasional Pernapasan Anak"],
  ],
  "Jawa Barat & DKI Jakarta": [
    ["RSCM Jakarta", "0211500135", "RS Rujukan Nasional & Anak"],
    ["RSUP Dr. Hasan Sadikin Bandung", "0222034953", "RS Rujukan Utama Jawa Barat"],
    ["RSUD Bandung Kiwari", "0225200505", "Khusus Ibu & Anak Kota Bandung"],
    ["Puskesmas Cimenyan", "0222503619", "Layanan Kesehatan Kec. Cimenyan"],
  ],
  "Aceh & Sumatera Utara": [
    ["RSUD Dr. Zainoel Abidin", "065134565", "RS Rujukan Utama Aceh"],
    ["RSUD Simeulue (3T)", "065021018", "IGD & Poli Anak Simeulue"],
    ["RSUP H. Adam Malik Medan", "0618360381", "RS Rujukan Utama Sumut"],
    ["RSUD dr. M. Thomsen Nias", "063921018", "Rujukan Kepulauan Nias (3T)"],
  ],
  "Sumatera Barat, Riau & Kepri": [
    ["RSUP Dr. M. Djamil Padang", "075132371", "RS Rujukan Utama Sumbar"],
    ["RSUD Mentawai (Tuapejat)", "0759320018", "Rujukan Kepulauan Mentawai (3T)"],
    ["RSUD Arifin Achmad Pekanbaru", "076121618", "RS Rujukan Utama Riau"],
    ["RSUD Kabupaten Natuna", "077331018", "RS Perbatasan Natuna (3T)"],
  ],
  "Jawa Tengah, DIY & Jawa Timur": [
    ["RSUP Dr. Kariadi Semarang", "0248413476", "RS Rujukan Utama Jateng"],
    ["RSUP Dr. Sardjito Yogyakarta", "0274587333", "RS Rujukan Utama DIY"],
    ["RSUD Dr. Soetomo Surabaya", "0315501078", "RS Rujukan Utama Jatim"],
    ["RSUD Moh. Anwar Sumenep", "0328662494", "Rujukan Kepulauan Madura"],
  ],
  "Bali, NTB & NTT": [
    ["RSUP Prof. Ngoerah Denpasar", "0361227911", "RS Rujukan Utama Bali"],
    ["RSUD Provinsi NTB Mataram", "03707502424", "RS Rujukan Utama NTB"],
    ["RSUD Prof. W.Z. Johannes Kupang", "0380833614", "RS Rujukan Utama NTT"],
    ["RSUD Umbu Rara Meha Sumba", "038761302", "RS Rujukan Pulau Sumba (3T)"],
  ],
  "Kalimantan": [
    ["RSUD Dr. Soedarso Pontianak", "0561737701", "RS Rujukan Utama Kalbar"],
    ["RSUD Putussibau (Kapuas Hulu)", "056721018", "Perbatasan Kalbar (3T)"],
    ["RSUD Ulin Banjarmasin", "05113252180", "RS Rujukan Utama Kalsel"],
    ["RSUD Kabupaten Nunukan", "055621118", "RS Perbatasan Kaltara (3T)"],
  ],
  "Sulawesi, Maluku & Papua": [
    ["RSUP Dr. Wahidin Makassar", "0411584677", "Rujukan Utama Indonesia Timur"],
    ["RSUP Prof. R.D. Kandou Manado", "0431838305", "RS Rujukan Utama Sulut"],
    ["RSUD Dr. M. Haulussy Ambon", "0911344871", "RS Rujukan Utama Maluku"],
    ["RSUD Dok II Jayapura", "0967533616", "RS Rujukan Utama Papua"],
    ["RSUD Wamena (Jayawijaya)", "096931066", "Papua Pegunungan (3T)"],
  ],
};

let currentAudioSamples = null;
let currentSampleRate = 22050;
let isRecording = false;
let mediaStream = null;
let audioCtx = null;
let recorderNode = null;
let recordedChunks = [];
let latestReportText = "";

window.onload = () => {
  const sel = document.getElementById("regionSelect");
  Object.keys(OFFLINE_FASKES_DB).forEach(r => {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = r;
    sel.appendChild(opt);
  });
};

function goStep(n) {
  document.querySelectorAll(".step").forEach(el => el.classList.remove("active"));
  document.getElementById("step" + n).classList.add("active");
  window.scrollTo(0, 0);
}

async function toggleRecording() {
  const btn = document.getElementById("btnRecord");
  const status = document.getElementById("audioStatus");
  if (!isRecording) {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(mediaStream);
      recorderNode = audioCtx.createScriptProcessor(4096, 1, 1);
      recordedChunks = [];
      recorderNode.onaudioprocess = e => {
        recordedChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      };
      source.connect(recorderNode);
      recorderNode.connect(audioCtx.destination);
      isRecording = true;
      btn.textContent = "Hentikan Rekaman (Sedang Merekam...)";
      btn.className = "btn btn-danger";
      status.textContent = "Merekam suara batuk... Dekatkan HP 15-20 cm.";
    } catch (err) {
      alert("Izin mikrofon ditolak atau tidak tersedia: " + err.message);
    }
  } else {
    recorderNode.disconnect();
    mediaStream.getTracks().forEach(t => t.stop());
    const totalLen = recordedChunks.reduce((acc, c) => acc + c.length, 0);
    const merged = new Float32Array(totalLen);
    let offset = 0;
    for (const chunk of recordedChunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    currentAudioSamples = merged;
    currentSampleRate = audioCtx.sampleRate;
    isRecording = false;
    btn.textContent = "Rekam Ulang Suara Batuk";
    btn.className = "btn btn-warn";
    status.textContent = `Suara terekam (${(totalLen / currentSampleRate).toFixed(1)} detik). Siap dianalisis!`;
  }
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async e => {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuf = await ctx.decodeAudioData(e.target.result);
      currentAudioSamples = audioBuf.getChannelData(0);
      currentSampleRate = audioBuf.sampleRate;
      document.getElementById("audioStatus").textContent = `File dipilih: ${file.name} (${audioBuf.duration.toFixed(1)} dtk)`;
    } catch (err) {
      alert("Gagal membaca file audio: " + err.message);
    }
  };
  reader.readAsArrayBuffer(file);
}

function fftRadix2(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wlenRe = Math.cos(ang), wlenIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1, wIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j], uIm = im[i + j];
        const vRe = re[i + j + len / 2] * wRe - im[i + j + len / 2] * wIm;
        const vIm = re[i + j + len / 2] * wIm + im[i + j + len / 2] * wRe;
        re[i + j] = uRe + vRe; im[i + j] = uIm + vIm;
        re[i + j + len / 2] = uRe - vRe; im[i + j + len / 2] = uIm - vIm;
        const nextWRe = wRe * wlenRe - wIm * wlenIm;
        wIm = wRe * wlenIm + wIm * wlenRe;
        wRe = nextWRe;
      }
    }
  }
}

function extractMelSpectrogram128x128(samples, sr) {
  const targetSr = 22050;
  const targetLen = Math.floor((samples.length / sr) * targetSr);
  const y = new Float32Array(Math.min(targetLen, targetSr * 3));
  for (let i = 0; i < y.length; i++) {
    const idx = (i / targetSr) * sr;
    const i0 = Math.floor(idx);
    const i1 = Math.min(i0 + 1, samples.length - 1);
    const frac = idx - i0;
    y[i] = samples[i0] * (1 - frac) + samples[i1] * frac;
  }

  let sumSq = 0, peak = 0;
  for (let i = 0; i < y.length; i++) {
    sumSq += y[i] * y[i];
    if (Math.abs(y[i]) > peak) peak = Math.abs(y[i]);
  }
  const rms = Math.sqrt(sumSq / Math.max(1, y.length));
  if (rms < 0.004) throw new Error("Suara terlalu pelan atau hening. Dekatkan HP 15-20 cm saat anak batuk.");
  if (peak > 0) for (let i = 0; i < y.length; i++) y[i] /= peak;

  const nFft = 2048, hop = 512, nMels = 128;
  const win = new Float32Array(nFft);
  for (let i = 0; i < nFft; i++) win[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (nFft - 1)));

  const numFrames = Math.max(1, Math.floor((y.length - nFft) / hop) + 1);
  const halfFft = nFft / 2 + 1;
  const powerSpec = Array.from({ length: halfFft }, () => new Float32Array(numFrames));

  const re = new Float32Array(nFft), im = new Float32Array(nFft);
  for (let f = 0; f < numFrames; f++) {
    re.fill(0); im.fill(0);
    const start = f * hop;
    for (let i = 0; i < nFft && start + i < y.length; i++) re[i] = y[start + i] * win[i];
    fftRadix2(re, im);
    for (let k = 0; k < halfFft; k++) powerSpec[k][f] = re[k] * re[k] + im[k] * im[k];
  }

  const melMin = 0, melMax = 2595 * Math.log10(1 + (targetSr / 2) / 700);
  const bins = [];
  for (let m = 0; m < nMels + 2; m++) {
    const mel = melMin + (m / (nMels + 1)) * (melMax - melMin);
    const hz = 700 * (Math.pow(10, mel / 2595) - 1);
    bins.push(Math.floor(((nFft + 1) * hz) / targetSr));
  }

  const melDb = Array.from({ length: nMels }, () => new Float32Array(numFrames));
  let maxDb = -1e9, minDb = 1e9;
  for (let m = 1; m <= nMels; m++) {
    for (let f = 0; f < numFrames; f++) {
      let sum = 0;
      for (let k = bins[m - 1]; k < bins[m]; k++) {
        if (bins[m] !== bins[m - 1]) sum += ((k - bins[m - 1]) / (bins[m] - bins[m - 1])) * powerSpec[k][f];
      }
      for (let k = bins[m]; k < bins[m + 1]; k++) {
        if (bins[m + 1] !== bins[m]) sum += ((bins[m + 1] - k) / (bins[m + 1] - bins[m])) * powerSpec[k][f];
      }
      const db = 10 * Math.log10(Math.max(sum, 1e-10));
      melDb[m - 1][f] = db;
      if (db > maxDb) maxDb = db;
      if (db < minDb) minDb = db;
    }
  }

  const img = new Float32Array(128 * 128);
  const span = Math.max(1e-6, maxDb - minDb);
  for (let r = 0; r < 128; r++) {
    const srcR = Math.min(nMels - 1, Math.floor((r / 127) * (nMels - 1)));
    for (let c = 0; c < 128; c++) {
      const srcC = Math.min(numFrames - 1, Math.floor((c / 127) * (numFrames - 1)));
      img[r * 128 + c] = (melDb[srcR][srcC] - minDb) / span;
    }
  }
  return img;
}

function conv2dBnPool(input, H, W, cIn, cOut, kW, kB, gamma, beta, mean, varArr) {
  const convOut = new Float32Array(H * W * cOut);
  const invStd = new Float32Array(cOut);
  for (let oc = 0; oc < cOut; oc++) invStd[oc] = gamma[oc] / Math.sqrt(varArr[oc] + 1e-3);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const outBase = (y * W + x) * cOut;
      for (let oc = 0; oc < cOut; oc++) {
        let sum = kB[oc];
        for (let ky = 0; ky < 3; ky++) {
          const iy = y + ky - 1;
          if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < 3; kx++) {
            const ix = x + kx - 1;
            if (ix < 0 || ix >= W) continue;
            const inBase = (iy * W + ix) * cIn;
            for (let ic = 0; ic < cIn; ic++) {
              sum += input[inBase + ic] * kW[ky][kx][ic][oc];
            }
          }
        }
        const reluVal = sum > 0 ? sum : 0;
        convOut[outBase + oc] = (reluVal - mean[oc]) * invStd[oc] + beta[oc];
      }
    }
  }

  const pH = H >> 1, pW = W >> 1;
  const poolOut = new Float32Array(pH * pW * cOut);
  for (let y = 0; y < pH; y++) {
    for (let x = 0; x < pW; x++) {
      const dstBase = (y * pW + x) * cOut;
      for (let oc = 0; oc < cOut; oc++) {
        let m = -1e9;
        for (let dy = 0; dy < 2; dy++) {
          for (let dx = 0; dx < 2; dx++) {
            const v = convOut[((y * 2 + dy) * W + (x * 2 + dx)) * cOut + oc];
            if (v > m) m = v;
          }
        }
        poolOut[dstBase + oc] = m;
      }
    }
  }
  return poolOut;
}

function predictCNN(img128) {
  const W = RESPIKIDD_WEIGHTS;
  let x = conv2dBnPool(img128, 128, 128, 1, 16, W.layer_0_Conv2D_w0, W.layer_0_Conv2D_w1,
    W.layer_1_BatchNormalization_w0, W.layer_1_BatchNormalization_w1, W.layer_1_BatchNormalization_w2, W.layer_1_BatchNormalization_w3);
  x = conv2dBnPool(x, 64, 64, 16, 32, W.layer_3_Conv2D_w0, W.layer_3_Conv2D_w1,
    W.layer_4_BatchNormalization_w0, W.layer_4_BatchNormalization_w1, W.layer_4_BatchNormalization_w2, W.layer_4_BatchNormalization_w3);
  x = conv2dBnPool(x, 32, 32, 32, 64, W.layer_6_Conv2D_w0, W.layer_6_Conv2D_w1,
    W.layer_7_BatchNormalization_w0, W.layer_7_BatchNormalization_w1, W.layer_7_BatchNormalization_w2, W.layer_7_BatchNormalization_w3);

  const gap = new Float32Array(64);
  for (let i = 0; i < 256; i++) {
    for (let c = 0; c < 64; c++) gap[c] += x[i * 64 + c];
  }
  for (let c = 0; c < 64; c++) gap[c] /= 256;

  const d1 = new Float32Array(64);
  for (let j = 0; j < 64; j++) {
    let s = W.layer_11_Dense_w1[j];
    for (let i = 0; i < 64; i++) s += gap[i] * W.layer_11_Dense_w0[i][j];
    d1[j] = s > 0 ? s : 0;
  }

  let logit = W.layer_12_Dense_w1[0];
  for (let i = 0; i < 64; i++) logit += d1[i] * W.layer_12_Dense_w0[i][0];
  return 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, logit))));
}

function runOfflineAnalysis() {
  const child = document.getElementById("childName").value.trim();
  const parent = document.getElementById("parentName").value.trim();
  if (!child || !parent) return alert("Mohon isi Nama Anak dan Nama Orang Tua terlebih dahulu.");
  if (!currentAudioSamples) return alert("Silakan rekam suara batuk atau pilih file .wav terlebih dahulu.");

  try {
    const t0 = performance.now();
    const img = extractMelSpectrogram128x128(currentAudioSamples, currentSampleRate);
    const rawProb = predictCNN(img);
    let score = 1.0 - rawProb;
    if (document.getElementById("chkWheezing").checked) score = Math.min(1.0, score + 0.05);
    if (document.getElementById("chkFever").checked) score = Math.min(1.0, score + 0.03);
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

    const pct = (score * 100).toFixed(1);
    const isHigh = score >= 0.5;
    const resCard = document.getElementById("resultCard");
    resCard.className = "card " + (isHigh ? "red" : "green");
    document.getElementById("resTitle").textContent = isHigh ? "TERDETEKSI INDIKASI GEJALA BRONKITIS" : "TIDAK TERDETEKSI GEJALA BRONKITIS";
    document.getElementById("resTitle").style.color = isHigh ? "#BE123C" : "#047857";
    document.getElementById("resScore").textContent = `Skor Risiko AI: ${pct}% (Waktu proses: ${elapsed} dtk)`;

    const age = document.getElementById("childAge").value || "-";
    const region = document.getElementById("regionSelect").value;
    const village = document.getElementById("villageInput").value || "-";
    const nowStr = new Date().toLocaleString("id-ID");

    const infoStr = `Waktu: ${nowStr}\nNama Anak: ${child} (${age} bln)\nOrang Tua: ${parent}\nWilayah: ${region} (${village})`;
    document.getElementById("patientSummary").textContent = infoStr;
    latestReportText = `=== LAPORAN SKRINING RESPIKIDD (OFFLINE) ===\n${infoStr}\nHasil AI: ${document.getElementById("resTitle").textContent} (${pct}%)\n`;

    const faskesDiv = document.getElementById("faskesList");
    faskesDiv.innerHTML = "";
    const list = [...(OFFLINE_FASKES_DB[region] || []), ...OFFLINE_FASKES_DB["Layanan Darurat Nasional (24 Jam)"]];
    list.slice(0, 5).forEach(([nama, telp, ket]) => {
      const div = document.createElement("div");
      div.className = "faskes-item";
      div.innerHTML = `<b>${nama}</b><br><span style="color:#64748B">${ket}</span><br><a class="call-link" href="tel:${telp}">Hubungi: ${telp}</a>`;
      faskesDiv.appendChild(div);
    });

    goStep(4);
  } catch (err) {
    alert(err.message);
  }
}

function downloadReport() {
  const blob = new Blob([latestReportText], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `Laporan_RespiKidd_${Date.now()}.txt`;
  a.click();
}