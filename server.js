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

// ---- sharp: ใช้ค่านิ่งๆ ที่เสถียร ----
sharp.cache(true);
sharp.concurrency(1);

const app = express();
const PORT = process.env.PORT || 3000;
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;

// ===== Config (คงแนวเดิม + จูนความมั่นใจ) =====
const INPUT_SIZE = Number(process.env.INPUT_SIZE || 300);

// ⚠️ โมเดลตอนเทรนมี layers.Rescaling(1./255) อยู่แล้ว
//    เพราะฉะนั้น *อย่า* หาร 255 ซ้ำที่ inference
const MODEL_INCLUDES_RESCALE = true;

// Unknown policy (สมดุล)
const UNKNOWN_THRESHOLD = Number(process.env.UNKNOWN_THRESHOLD || 0.55);
const MARGIN_THRESHOLD  = Number(process.env.MARGIN_THRESHOLD  || 0.08);
const ENTROPY_THRESHOLD = Number(process.env.ENTROPY_THRESHOLD || 1.60);

// Sharpen (ยกกำลัง probs หลัง softmax) — คมขึ้นแบบพอดี
const PROB_SHARPEN_GAMMA = Number(process.env.PROB_SHARPEN_GAMMA || 1.36);

// (ทางเลือก) คาลิเบรตน้ำหนักคลาสหลัง softmax
// ตัวอย่าง ENV: CALIB_WEIGHTS=Eczema:1.45,Shingles:1.40
// ตัวอย่าง ENV: PER_CLASS_THRESHOLDS=Eczema:0.48,Shingles:0.48
function parseMapEnv(str) {
  const map = {};
  if (!str) return map;
  for (const tok of String(str).split(",")) {
    const [k, v] = tok.split(":").map(s => s?.trim());
    if (k && v && !Number.isNaN(Number(v))) map[k] = Number(v);
  }
  return map;
}
const CALIB_WEIGHTS        = parseMapEnv(process.env.CALIB_WEIGHTS);
const PER_CLASS_THRESHOLDS = parseMapEnv(process.env.PER_CLASS_THRESHOLDS);

// ===== 1) โหลด labels + model =====
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
      // โมเดลเดิม export เป็น GraphModel (จาก tfjs-converter)
      model = await tf.loadGraphModel(MODEL_PATH);
      modelType = "graph";
      modelReady = true;
      console.log("✅ TFJS GraphModel loaded");
    } catch {
      // เผื่อไว้กรณีเป็น Keras LayersModel
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

// ===== Helper: ตอบ LINE =====
async function replyMessage(replyToken, text) {
  if (!replyToken) return;
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

// ===== Utils =====
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
    s += x;
    return x;
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

// ===== 2) Preprocess + Predict + Unknown policy =====
async function classifyImage(imageBuffer, { debug = false } = {}) {
  if (!modelReady) throw new Error("Model not ready");

  // 1) Preprocess (แบบเรียบง่ายเหมือนเดิม แต่ robust ขึ้นเล็กน้อย)
  const resized = await sharp(imageBuffer, { limitInputPixels: false })
    .rotate() // เคารพ EXIF
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
    .toFormat("png")
    .toBuffer();

  // decode -> float32 -> (หาร 255 เฉพาะถ้าโมเดล *ไม่มี* Rescaling)
  let x = tf.node.decodeImage(resized, 3).toFloat().expandDims(0);
  if (!MODEL_INCLUDES_RESCALE) x = x.div(255);

  // 2) Predict (รองรับทั้ง Graph & Layers)
  let y = model.predict ? model.predict(x) : null;
  if (Array.isArray(y)) y = y[0];
  if (!y || typeof y.dataSync !== "function") {
    try {
      const feedName  = model.inputs?.[0]?.name;
      const fetchName = model.outputs?.[0]?.name;
      y = model.execute(feedName ? { [feedName]: x } : x, fetchName);
    } catch (e) {
      tf.dispose(x);
      throw e;
    }
  }

  // 3) Post-process -> probs
  let probs = Array.from(y.dataSync());
  const sum = probs.reduce((a, b) => a + b, 0);
  if (!Number.isFinite(sum) || Math.abs(sum - 1) > 1e-3 || probs.some(v => v < 0) || probs.some(v => v > 1)) {
    probs = softmax(probs);
  }

  // 4) (ใหม่) คาลิเบรตเฉพาะคลาสที่เพี้ยน เช่น Eczema, Shingles
  probs = applyCalibration(probs, labels, CALIB_WEIGHTS);

  // 5) sharpen ให้ top-1 มั่นใจกว่าเดิมเล็กน้อย (ไม่โอเวอร์)
  probs = sharpen(probs, PROB_SHARPEN_GAMMA);

  // 6) Unknown policy
  const { bestIdx, bestProb, secondProb } = top2(probs);
  const ent = entropy(probs);

  // per-class threshold (ถ้าตั้งมาใน ENV) เช่น Eczema:0.48, Shingles:0.48
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
    console.log("[DEBUG] probs:", probs.map(v => v.toFixed(4)));
    console.log(`[DEBUG] best=${bestName} p=${bestProb.toFixed(4)} second=${secondProb.toFixed(4)} H=${ent.toFixed(4)} thr=${thrClass}`);
    console.log(`[DEBUG] unknown=${isUnknown}`);
  }

  tf.dispose([x, y]);
  return { label, score, appliedUnknown: isUnknown };
}

// ===== 3) Webhook =====
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
      // เก็บรายละเอียดไว้ใน log แต่ตอบผู้ใช้สั้นๆ
      console.error("[Webhook error]", err?.stack || err?.message || err);
      await replyMessage(replyToken, "เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้งค่ะ");
    }
  }
  res.sendStatus(200);
});

// ===== 4) Health & Debug =====
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

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
