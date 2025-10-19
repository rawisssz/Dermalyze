// ---- ลด log TF (ต้องมาก่อน tf) ----
process.env.TF_CPP_MIN_LOG_LEVEL = process.env.TF_CPP_MIN_LOG_LEVEL || "2";

require("dotenv").config();
const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

sharp.cache(true);
sharp.concurrency(1);

const app = express();
const PORT = process.env.PORT || 3000;
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;

// ======= Config =======
const INPUT_SIZE = Number(process.env.INPUT_SIZE || 300);
// โมเดลตอนเทรนมี Rescaling(1./255) แล้ว → ห้ามหาร 255 ซ้ำ
const MODEL_INCLUDES_RESCALE = true;

// Unknown policy
const UNKNOWN_THRESHOLD = Number(process.env.UNKNOWN_THRESHOLD || 0.55);
const MARGIN_THRESHOLD  = Number(process.env.MARGIN_THRESHOLD  || 0.08);
const ENTROPY_THRESHOLD = Number(process.env.ENTROPY_THRESHOLD || 1.60);

// sharpen probs (ยกกำลังแล้ว normalize)
const PROB_SHARPEN_GAMMA = Number(process.env.PROB_SHARPEN_GAMMA || 1.36);

// ==== Calibration / per-class thresholds (อ่านจาก ENV + default ที่จูนไว้) ====
function parseMapEnv(str) {
  const map = {};
  if (!str) return map;
  for (const tok of String(str).split(",")) {
    const [k, v] = tok.split(":").map(s => s?.trim());
    if (k && v && !Number.isNaN(Number(v))) map[k] = Number(v);
  }
  return map;
}
let CALIB_WEIGHTS = parseMapEnv(process.env.CALIB_WEIGHTS);
let PER_CLASS_THRESHOLDS = parseMapEnv(process.env.PER_CLASS_THRESHOLDS);

// ======= Load labels + model =======
const MODEL_DIR   = path.join(__dirname, "model");
const MODEL_PATH  = `file://${path.join(MODEL_DIR, "model.json")}`;
const LABELS_PATH = path.join(__dirname, "class_names.json");

let labels = [];
try {
  labels = JSON.parse(fs.readFileSync(LABELS_PATH, "utf-8"));
  if (!Array.isArray(labels) || labels.length < 2) throw new Error("labels invalid");
  console.log("✅ Loaded labels:", labels);
} catch (e) {
  console.error("❌ Load labels failed:", e.message);
  labels = ["ClassA", "ClassB", "Unknown"];
}
const UNKNOWN_LABEL_INDEX = labels.length - 1;

let model = null, modelReady = false, modelType = "unknown";
(async () => {
  try {
    try {
      model = await tf.loadGraphModel(MODEL_PATH);
      modelType = "graph";
      modelReady = true;
      console.log("✅ TFJS GraphModel loaded");
    } catch {
      model = await tf.loadLayersModel(MODEL_PATH);
      modelType = "layers";
      modelReady = true;
      console.log("✅ TFJS LayersModel loaded");
    }
  } catch (err) {
    console.error("❌ Failed to load model:", err);
  }
})();

app.use(bodyParser.json());

// ======= Helpers =======
async function replyMessage(replyToken, text) {
  try {
    await axios.post(
      "https://api.line.me/v2/bot/message/reply",
      { replyToken, messages: [{ type: "text", text }] },
      { headers: { Authorization: `Bearer ${LINE_ACCESS_TOKEN}` }, timeout: 15000 }
    );
  } catch (e) {
    console.error("Reply error:", e?.response?.data || e.message);
  }
}

function entropy(ps) {
  let h = 0;
  for (const p of ps) if (p > 0) h -= p * Math.log(p);
  return h;
}
function top2(ps) {
  let b = [-1, -1], s = [-1, -1];
  for (let i = 0; i < ps.length; i++) {
    const p = ps[i];
    if (p > b[1]) { s = b; b = [i, p]; }
    else if (p > s[1]) { s = [i, p]; }
  }
  return { bestIdx: b[0], bestProb: b[1], secondIdx: s[0], secondProb: s[1] };
}
function softmax(arr) {
  const m = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / (s || 1));
}
function normalize(arr) {
  let s = 0;
  const clipped = arr.map(v => {
    const x = Number.isFinite(v) ? Math.max(0, v) : 0;
    s += x; return x;
  });
  if (s <= 0) return Array(arr.length).fill(1 / arr.length);
  return clipped.map(v => v / s);
}
function sharpen(probs, gamma) {
  if (!Number.isFinite(gamma) || gamma <= 0) return probs;
  return normalize(probs.map(p => Math.pow(Math.max(p, 1e-12), gamma)));
}
function applyCalibration(probs, labels, weightMap) {
  if (!weightMap || !Object.keys(weightMap).length) return probs;
  const scaled = probs.map((p, i) => {
    const name = labels[i] || "";
    const w = Number.isFinite(weightMap[name]) ? weightMap[name] : 1.0;
    return p * w;
  });
  return normalize(scaled);
}

// =======  Inference (พร้อม TTA 3 วิว)  =======
async function classifyImage(imageBuffer, { debug = false } = {}) {
  if (!modelReady) throw new Error("Model not ready");

  // อ่าน/ปรับขนาด (cover) + เคารพ EXIF
  const basePNG = await sharp(imageBuffer, { limitInputPixels: false })
    .rotate()
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
    .toFormat("png")
    .toBuffer();

  // x0 : ภาพปกติ
  let x0 = tf.node.decodeImage(basePNG, 3).toFloat().expandDims(0);
  if (!MODEL_INCLUDES_RESCALE) x0 = x0.div(255);

  // x1 : กลับซ้าย–ขวา
  let x1 = tf.image.flipLeftRight(x0);

  // x2 : เพิ่มคอนทราสต์เล็กน้อย (บนสเกล 0..255)
  let x2 = tf.image.adjustContrast(x0, 1.12);
  x2 = tf.clipByValue(x2, 0, 255);

  // รวมเป็น batch 3 ภาพ
  const xb = tf.concat([x0, x1, x2], 0);

  let y = model.predict ? model.predict(xb) : null;
  if (Array.isArray(y)) y = y[0];
  if (!y || typeof y.dataSync !== "function") {
    try {
      const feedName  = model.inputs?.[0]?.name;
      const fetchName = model.outputs?.[0]?.name;
      y = model.execute(feedName ? { [feedName]: xb } : xb, fetchName);
    } catch (e) {
      tf.dispose([x0, x1, x2, xb]);
      throw e;
    }
  }

  // y: [3, C] → average (logits หรือ probs ก็รองรับ)
  const raw = y.arraySync(); // [[...C], [...], [...]]
  let probsAvg = Array(raw[0].length).fill(0);
  for (let i = 0; i < raw.length; i++) {
    const vec = raw[i];
    const needSoftmax = (Math.abs(vec.reduce((a,b)=>a+b,0) - 1) > 1e-3) || vec.some(v => v < 0) || vec.some(v => v > 1);
    const p = needSoftmax ? softmax(vec) : vec.slice();
    for (let c = 0; c < p.length; c++) probsAvg[c] += p[c];
  }
  probsAvg = probsAvg.map(v => v / raw.length);

  // ---- default calibration (ถ้าไม่ได้ตั้ง ENV) ----
  if (!Object.keys(CALIB_WEIGHTS).length) {
    CALIB_WEIGHTS = { Eczema: 1.55, Shingles: 1.45 };
  }
  if (!Object.keys(PER_CLASS_THRESHOLDS).length) {
    PER_CLASS_THRESHOLDS = { Eczema: 0.48, Shingles: 0.48 };
  }

  // 1) คาลิเบรตเฉพาะคลาส (เช่น Eczema/Shingles)
  probsAvg = applyCalibration(probsAvg, labels, CALIB_WEIGHTS);

  // 2) sharpen ให้ top-1 มั่นใจขึ้นแบบพอดี
  probsAvg = sharpen(probsAvg, PROB_SHARPEN_GAMMA);

  // 3) Unknown policy (รองรับ per-class threshold)
  const { bestIdx, bestProb, secondProb } = top2(probsAvg);
  const ent = entropy(probsAvg);

  const bestName = labels[bestIdx] || "";
  const thrClass = Number.isFinite(PER_CLASS_THRESHOLDS[bestName])
    ? PER_CLASS_THRESHOLDS[bestName]
    : UNKNOWN_THRESHOLD;

  const isUnknown =
    (bestProb < thrClass) ||
    (bestProb - secondProb < MARGIN_THRESHOLD) ||
    (ent > ENTROPY_THRESHOLD);

  const idx = isUnknown ? UNKNOWN_LABEL_INDEX : bestIdx;
  const label = labels[idx] || "ไม่สามารถจำแนกได้";
  const score = Number((bestProb * 100).toFixed(2));

  if (debug) {
    console.log("[DEBUG] probs:", probsAvg.map(v => v.toFixed(4)));
    console.log(`[DEBUG] best=${bestName} p=${bestProb.toFixed(4)} second=${secondProb.toFixed(4)} H=${ent.toFixed(4)} thr=${thrClass}`);
    console.log(`[DEBUG] unknown=${isUnknown}`);
  }

  tf.dispose([x0, x1, x2, xb, y]);
  return { label, score, appliedUnknown: isUnknown };
}

// ======= Webhook =======
app.post("/webhook", async (req, res) => {
  const events = req.body?.events || [];
  for (const event of events) {
    const replyToken = event.replyToken;
    try {
      if (!modelReady) {
        await replyMessage(replyToken, "โมเดลกำลังโหลดอยู่ กรุณาลองอีกครั้งในไม่กี่วินาทีค่ะ");
        continue;
      }

      if (event.type === "message" && event.message.type === "image") {
        const imageId = event.message.id;
        const imgResp = await axios.get(
          `https://api-data.line.me/v2/bot/message/${imageId}/content`,
          {
            headers: { Authorization: `Bearer ${LINE_ACCESS_TOKEN}` },
            responseType: "arraybuffer",
            timeout: 20000,
          }
        );

        const { label, score, appliedUnknown } = await classifyImage(imgResp.data);
        const extra = appliedUnknown ? " (จัดเป็น Unknown)" : "";
        await replyMessage(
          replyToken,
          `ผลการจำแนก: ${label}${extra}\nความเชื่อมั่นของคลาสสูงสุด ~${score}%`
        );
      } else if (event.type === "message" && event.message.type === "text") {
        await replyMessage(replyToken, "ส่งรูปมาเพื่อให้ช่วยจำแนกโรคผิวหนังได้เลยค่ะ");
      } else {
        await replyMessage(replyToken, "ยังรองรับเฉพาะรูปภาพและข้อความนะคะ");
      }
    } catch (err) {
      console.error("[Webhook error]", err?.stack || err?.message || err);
      await replyMessage(replyToken, "เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้งค่ะ");
    }
  }
  res.sendStatus(200);
});

// ======= Health & Debug =======
app.get("/", (_req, res) => res.send("Webhook is working!"));
app.get("/healthz", (_req, res) =>
  res.json({
    ok: true,
    modelReady,
    modelType,
    nLabels: labels.length,
    labels,
    thresholds: { UNKNOWN_THRESHOLD, MARGIN_THRESHOLD, ENTROPY_THRESHOLD, PROB_SHARPEN_GAMMA },
    calib: CALIB_WEIGHTS,
    perClassThresholds: PER_CLASS_THRESHOLDS,
    inputSize: INPUT_SIZE
  })
);

app.post("/debug/classify", express.raw({ type: "*/*", limit: "10mb" }), async (req, res) => {
  try {
    const out = await classifyImage(req.body, { debug: true });
    res.json(out);
  } catch (e) {
    console.error("[debug/classify]", e?.stack || e?.message || e);
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
