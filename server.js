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

// ===== Config =====
const INPUT_SIZE = Number(process.env.INPUT_SIZE || 300);
const MODEL_INCLUDES_RESCALE = true; // โมเดลมี Rescaling(1./255) แล้ว

// Unknown policy (global)
const UNKNOWN_THRESHOLD = Number(process.env.UNKNOWN_THRESHOLD || 0.55);
const MARGIN_THRESHOLD  = Number(process.env.MARGIN_THRESHOLD  || 0.08);
const ENTROPY_THRESHOLD = Number(process.env.ENTROPY_THRESHOLD || 1.60);

// Sharpen ความน่าจะเป็น (หลังรวม TTA)
const PROB_SHARPEN_GAMMA = Number(process.env.PROB_SHARPEN_GAMMA || 1.6);

// Calibration เฉพาะคลาส (ผ่าน env)
// ตัวอย่าง: CALIB_WEIGHTS="Eczema:1.45,Psoriasis:1.25,Shingles:1.40"
// ตัวอย่าง: PER_CLASS_THRESHOLDS="Eczema:0.48,Shingles:0.48"
function parseKV(raw) {
  const map = {};
  (raw || "").split(",").map(s => s.trim()).filter(Boolean).forEach(pair => {
    const [k, v] = pair.split(":").map(x => x.trim());
    if (k && v && !Number.isNaN(Number(v))) map[k] = Number(v);
  });
  return map;
}
const CALIB_WEIGHTS = parseKV(process.env.CALIB_WEIGHTS);
const PER_CLASS_THRESHOLDS = parseKV(process.env.PER_CLASS_THRESHOLDS);

// ===== 1) โหลด labels + model =====
const MODEL_DIR  = path.join(__dirname, "model");
const MODEL_PATH = `file://${path.join(MODEL_DIR, "model.json")}`;
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

let model = null, modelReady = false, modelType = "unknown";
(async () => {
  try {
    // ส่วนใหญ่ของเดียร์เป็น GraphModel
    model = await tf.loadGraphModel(MODEL_PATH);
    modelType = "graph";
    modelReady = true;
    console.log("✅ TFJS GraphModel loaded");
  } catch {
    try {
      model = await tf.loadLayersModel(MODEL_PATH);
      modelType = "layers";
      modelReady = true;
      console.log("✅ TFJS LayersModel loaded");
    } catch (err) {
      console.error("❌ Failed to load model:", err);
    }
  }
})();

app.use(bodyParser.json());

// ===== Helper: ตอบ LINE =====
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
function sharpenProbs(probs, gamma) {
  const raised = probs.map(p => Math.pow(Math.max(p, 1e-12), gamma));
  const s = raised.reduce((a, b) => a + b, 0);
  return raised.map(v => v / s);
}
function softmax(arr) {
  const m = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / s);
}
function applyClassWeights(probs, labels, weightMap) {
  const w = probs.map((p, i) => p * (weightMap[labels[i]] ?? 1));
  const s = w.reduce((a, b) => a + b, 0) || 1;
  return w.map(v => v / s);
}
function thresholdForClass(label) {
  return (PER_CLASS_THRESHOLDS && PER_CLASS_THRESHOLDS[label] != null)
    ? PER_CLASS_THRESHOLDS[label]
    : UNKNOWN_THRESHOLD;
}

// ===== TTA helpers (ใช้ tf.ops เพื่อไม่ต้องเขียนไฟล์ใหม่) =====
function ttaVariants(img4D) {
  // img4D: [1,H,W,3] float32 0..255 หรือ 0..1 (ขึ้นกับ MODEL_INCLUDES_RESCALE)
  const x = img4D;
  const xs = [];

  // 1) original
  xs.push(x);

  // 2) flip LR
  xs.push(tf.image.flipLeftRight(x));

  // 3) brighten up ~ +10%
  xs.push(tf.image.adjustBrightness(x, MODEL_INCLUDES_RESCALE ? 0.10 : 25.5));

  // 4) darken ~ -10%
  xs.push(tf.image.adjustBrightness(x, MODEL_INCLUDES_RESCALE ? -0.10 : -25.5));

  // 5) contrast 1.1
  const c = tf.image.adjustContrast(
    MODEL_INCLUDES_RESCALE ? x.div(1) : x.div(255), 1.1
  );
  xs.push(MODEL_INCLUDES_RESCALE ? c : c.mul(255));

  return xs;
}

function averageProbs(listOfProbs) {
  const n = listOfProbs.length;
  const sum = new Array(listOfProbs[0].length).fill(0);
  for (const ps of listOfProbs) for (let i = 0; i < ps.length; i++) sum[i] += ps[i];
  return sum.map(v => v / n);
}

// ===== 3) Preprocess + Predict + Unknown policy (with TTA) =====
async function classifyImage(imageBuffer, { debug = false } = {}) {
  if (!modelReady) throw new Error("Model not ready");

  const resized = await sharp(imageBuffer, { limitInputPixels: false })
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
    .toFormat("png")
    .toBuffer();

  let x = tf.node.decodeImage(resized, 3).toFloat().expandDims(0);
  if (!MODEL_INCLUDES_RESCALE) x = x; else x = x; // โมเดลมี Rescale ภายในแล้ว

  // ทำ TTA → ทำนายหลายครั้ง
  const variants = ttaVariants(x);
  const probsList = [];
  for (const v of variants) {
    let y = model.predict ? model.predict(v) : null;
    if (Array.isArray(y)) y = y[0];
    if (!y || typeof y.dataSync !== "function") {
      try {
        const feedName  = model.inputs?.[0]?.name;
        const fetchName = model.outputs?.[0]?.name;
        y = model.execute(feedName ? { [feedName]: v } : v, fetchName);
      } catch (e) {
        tf.dispose([x, ...variants]);
        throw e;
      }
    }

    let raw = Array.from(y.dataSync());
    const sum = raw.reduce((p, c) => p + c, 0);
    let probs = (Math.abs(sum - 1) > 1e-3) ? softmax(raw) : raw;
    probsList.push(probs);

    tf.dispose(y);
  }

  // รวมผล TTA
  let probs = averageProbs(probsList);

  // sharpen + class calibration
  probs = sharpenProbs(probs, PROB_SHARPEN_GAMMA);
  probs = applyClassWeights(probs, labels, CALIB_WEIGHTS);

  const { bestIdx, bestProb, secondProb } = top2(probs);
  const bestLabel = labels[bestIdx] || "Unknown";
  const ent = entropy(probs);

  const isUnknown =
    (bestProb < thresholdForClass(bestLabel)) ||
    (bestProb - secondProb < MARGIN_THRESHOLD) ||
    (ent > ENTROPY_THRESHOLD);

  const finalIdx = isUnknown ? (labels.length - 1) : bestIdx;
  const label = labels[finalIdx] || "Unknown";
  const score = Number((bestProb * 100).toFixed(2));

  if (debug) {
    console.log("[DEBUG] probs(TTA+sharpen+calib):", probs.map(v => v.toFixed(4)));
    console.log(`[DEBUG] best=${bestLabel} ${bestProb.toFixed(4)} second=${secondProb.toFixed(4)} H=${ent.toFixed(4)} thr=${thresholdForClass(bestLabel)}`);
  }

  tf.dispose([x, ...variants]);
  return { label, score, appliedUnknown: isUnknown };
}

// ===== 4) Webhook =====
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
      console.error("Webhook error:", err?.response?.data || err.message);
      await replyMessage(replyToken, "เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้งค่ะ");
    }
  }
  res.sendStatus(200);
});

// ===== 5) Health & Debug =====
app.get("/", (_req, res) => res.send("Webhook is working!"));
app.get("/healthz", (_req, res) => res.json({
  ok: true, modelReady, modelType,
  nLabels: labels.length,
  thresholds: {
    UNKNOWN_THRESHOLD, MARGIN_THRESHOLD, ENTROPY_THRESHOLD, PROB_SHARPEN_GAMMA,
    PER_CLASS_THRESHOLDS, CALIB_WEIGHTS
  }
}));
app.post("/debug/classify", express.raw({ type: "*/*", limit: "10mb" }), async (req, res) => {
  try {
    const out = await classifyImage(req.body, { debug: true });
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));