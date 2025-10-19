// ---- ลด TF logs (ต้องเซ็ตก่อน import tf) ----
process.env.TF_CPP_MIN_LOG_LEVEL = process.env.TF_CPP_MIN_LOG_LEVEL || "2";

require("dotenv").config();
const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

// ---- sharp safety ----
sharp.cache(true);
sharp.concurrency(1);

const app = express();
const PORT = process.env.PORT || 3000; // Render จะใส่ PORT ให้เอง
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;

// ---- Config (ปรับผ่าน ENV ได้) ----
const INPUT_SIZE = Math.max(32, Math.min(1024, Number(process.env.INPUT_SIZE || 300)));

const UNKNOWN_THRESHOLD = Number(process.env.UNKNOWN_THRESHOLD || 0.55);
const MARGIN_THRESHOLD = Number(process.env.MARGIN_THRESHOLD || 0.08);
const ENTROPY_THRESHOLD = Number(process.env.ENTROPY_THRESHOLD || 1.60);
const PROB_SHARPEN_GAMMA = Number(process.env.PROB_SHARPEN_GAMMA || 1.60);

// ex: "Eczema:1.45,Shingles:1.40"
function parseMapEnv(str) {
  const map = {};
  if (!str) return map;
  for (const tok of String(str).split(",")) {
    const [k, v] = tok.split(":").map(s => s?.trim());
    if (k && v && !Number.isNaN(Number(v))) map[k] = Number(v);
  }
  return map;
}
const CALIB_WEIGHTS = parseMapEnv(process.env.CALIB_WEIGHTS);               // scale per class
const PER_CLASS_THRESHOLDS = parseMapEnv(process.env.PER_CLASS_THRESHOLDS); // per-class unknown thresholds

// ======================================================
// 1) Load labels + model
// ======================================================
const MODEL_DIR = path.join(__dirname, "model");
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
const UNKNOWN_LABEL_INDEX = labels.length - 1;

let model = null;
let modelReady = false;
let modelType = "unknown"; // "layers" | "graph"

(async () => {
  try {
    try {
      model = await tf.loadLayersModel(MODEL_PATH);
      modelType = "layers";
      modelReady = true;
      console.log("✅ TFJS LayersModel loaded");
    } catch (e1) {
      console.warn("ℹ️ Not a LayersModel, trying GraphModel…");
      model = await tf.loadGraphModel(MODEL_PATH);
      modelType = "graph";
      modelReady = true;
      console.log("✅ TFJS GraphModel loaded");
    }
  } catch (err) {
    console.error("❌ Failed to load model:", err);
  }
})();

app.use(bodyParser.json());

// ======================================================
// 2) LINE reply helper
// ======================================================
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

// ======================================================
// 3) Utils
// ======================================================
function entropy(probArray) {
  let h = 0;
  for (const p of probArray) if (p > 0) h -= p * Math.log(p);
  return h;
}
function top2(probArray) {
  let bestIdx = -1, bestProb = -1;
  let secondIdx = -1, secondProb = -1;
  for (let i = 0; i < probArray.length; i++) {
    const p = probArray[i];
    if (p > bestProb) { secondIdx = bestIdx; secondProb = bestProb; bestIdx = i; bestProb = p; }
    else if (p > secondProb) { secondIdx = i; secondProb = p; }
  }
  return { bestIdx, bestProb, secondIdx, secondProb };
}

// ให้ probs รวมเป็น 1 เสมอ และไม่เป็น NaN
function normalizeProbs(arr) {
  let sum = 0;
  const safe = arr.map(v => {
    const x = Number.isFinite(v) ? Math.max(0, v) : 0;
    sum += x;
    return x;
  });
  if (sum <= 0) {
    // กระจายเท่ากัน
    const u = 1 / safe.length;
    return Array(safe.length).fill(u);
  }
  return safe.map(v => v / sum);
}

// Power transform (sharpen/soften) แล้ว normalize
function powSharpen(probs, gamma) {
  if (!Number.isFinite(gamma) || gamma <= 0) return probs;
  return normalizeProbs(probs.map(p => Math.pow(p, gamma)));
}

// Calibration โดย weight ต่อคลาส แล้ว normalize
function applyCalibration(probs, labels, weightMap) {
  if (!weightMap || !Object.keys(weightMap).length) return probs;
  const scaled = probs.map((p, i) => {
    const name = labels[i] || "";
    const w = Number.isFinite(weightMap[name]) ? weightMap[name] : 1.0;
    return p * w;
  });
  return normalizeProbs(scaled);
}

// ======================================================
// 4) Preprocess + Predict + Unknown policy
// ======================================================
async function classifyImage(imageBuffer, { debug = false } = {}) {
  if (!model || !modelReady) throw new Error("Model is not loaded yet");

  // --- preprocess (ปลอดภัยกับภาพทุกชนิด) ---
  const resized = await sharp(imageBuffer, { failOn: "none" })
    .rotate() // respect EXIF
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
    .toColorspace("rgb")
    .raw()
    .toBuffer({ resolveWithObject: true })
    .then(({ data }) => data);

  const x = tf.tensor(resized, [INPUT_SIZE, INPUT_SIZE, 3], "int32")
              .toFloat()
              .div(255)
              .expandDims(0); // [1,H,W,3]

  let out;
  try {
    out = model.predict ? model.predict(x) : null;
    if (Array.isArray(out)) out = out[0];

    if (!out || typeof out.dataSync !== "function") {
      // GraphModel execute ทางเลือก
      const feedName = model.inputs?.[0]?.name;
      const fetchName = model.outputs?.[0]?.name;
      if (!feedName) throw new Error("GraphModel feed tensor name not found");
      out = model.execute({ [feedName]: x }, fetchName);
    }
  } catch (e) {
    tf.dispose(x);
    throw e;
  }

  // ดึงค่าออกมา
  const raw = out.dataSync(); // Float32Array
  let probs = Array.from(raw);

  // ถ้ารวมไม่ใกล้ 1 ให้ softmax ใหม่ (แต่โมเดลเรามี softmax อยู่แล้ว โดยปกติ sums~1)
  const s = probs.reduce((p, c) => p + c, 0);
  if (!Number.isFinite(s) || Math.abs(s - 1) > 1e-3 || probs.some(v => v < 0) || probs.some(v => v > 1)) {
    // ทำ softmax แบบนิ่ง ๆ
    const m = Math.max(...probs);
    const exps = probs.map(v => Math.exp(v - m));
    const sumExp = exps.reduce((p, c) => p + c, 0);
    probs = exps.map(v => v / (sumExp || 1));
  }

  // ถ้าขนาดเอาต์พุตไม่เท่ากับ labels → ถือว่าไฟล์โมเดล/labels ไม่ตรงกัน
  if (probs.length !== labels.length) {
    tf.dispose([x, out]);
    throw new Error(`Model outputs ${probs.length} classes but labels has ${labels.length}`);
  }

  // 1) คาลิเบรต per-class (ดัน Eczema, Shingles ตาม ENV)
  probs = applyCalibration(probs, labels, CALIB_WEIGHTS);

  // 2) sharpen ด้วย gamma (>1 = คมขึ้น)
  probs = powSharpen(probs, PROB_SHARPEN_GAMMA);

  // 3) ตัดสินใจ Unknown ด้วย 3 เงื่อนไข OR (+ per-class thresholds)
  const { bestIdx, bestProb, secondProb } = top2(probs);
  const ent = entropy(probs);

  // per-class override
  const nameBest = labels[bestIdx] || "";
  const thrClass = Number.isFinite(PER_CLASS_THRESHOLDS[nameBest])
    ? PER_CLASS_THRESHOLDS[nameBest]
    : UNKNOWN_THRESHOLD;

  const unknown =
    (bestProb < thrClass) ||
    (bestProb - (secondProb ?? 0) < MARGIN_THRESHOLD) ||
    (ent > ENTROPY_THRESHOLD);

  const finalIdx = unknown ? UNKNOWN_LABEL_INDEX : bestIdx;
  const label = labels[finalIdx] || "ไม่สามารถจำแนกได้";
  const score = Number(((unknown ? bestProb : bestProb) * 100).toFixed(2));

  if (debug) {
    console.log("[DEBUG] probs:", probs.map(v => Number(v.toFixed(4))));
    console.log("[DEBUG] best:", nameBest, bestProb.toFixed(4), "second:", (secondProb ?? 0).toFixed(4), "entropy:", ent.toFixed(4));
    console.log("[DEBUG] thrClass:", thrClass, "unknown:", unknown);
  }

  tf.dispose([x, out]);
  return { label, score, appliedUnknown: unknown };
}

// ======================================================
// 5) Webhook
// ======================================================
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

        // ดึงรูปจาก LINE (retry เบาๆ)
        let imgResp = null;
        let lastErr = null;
        for (let attempt = 1; attempt <= 2; attempt++) {
          try {
            imgResp = await axios.get(
              `https://api-data.line.me/v2/bot/message/${imageId}/content`,
              {
                headers: { Authorization: `Bearer ${LINE_ACCESS_TOKEN}` },
                responseType: "arraybuffer",
                timeout: 20000,
              }
            );
            break;
          } catch (e) {
            lastErr = e;
            await new Promise(r => setTimeout(r, 200 * attempt));
          }
        }
        if (!imgResp) {
          console.error("[FETCH_IMAGE_ERROR]", {
            status: lastErr?.response?.status,
            data: lastErr?.response?.data?.toString?.()?.slice?.(0, 200),
            message: lastErr?.message
          });
          await replyMessage(replyToken, "ดึงรูปจาก LINE ไม่สำเร็จ กรุณาลองส่งใหม่อีกครั้งค่ะ");
          continue;
        }

        // จำแนก
        try {
          const { label, score, appliedUnknown } = await classifyImage(imgResp.data);
          const extra = appliedUnknown ? " (จัดเป็น Unknown)" : "";
          await replyMessage(
            replyToken,
            `ผลการจำแนก: ${label}${extra}\nความเชื่อมั่นของคลาสสูงสุด ~${score}%`
          );
        } catch (e) {
          console.error("[CLASSIFY_ERROR]", e?.stack || e?.message || e);
          await replyMessage(replyToken, "ประมวลผลภาพไม่สำเร็จ กรุณาลองใหม่อีกครั้งค่ะ (ลองส่งเป็น JPG/PNG ขนาดไม่ใหญ่เกินไป)");
        }

      } else if (event.type === "message" && event.message.type === "text") {
        await replyMessage(replyToken, "ส่งรูปมาเพื่อให้ช่วยจำแนกโรคผิวหนังได้เลยค่ะ");
      } else {
        await replyMessage(replyToken, "ยังรองรับเฉพาะรูปภาพและข้อความนะคะ");
      }
    } catch (err) {
      console.error("[WEBHOOK_TOPLEVEL_ERROR]", err?.stack || err?.message || err);
      await replyMessage(replyToken, "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์ ลองใหม่อีกครั้งค่ะ");
    }
  }
  res.sendStatus(200);
});

// ======================================================
// 6) Health & debug
// ======================================================
app.get("/", (_req, res) => res.send("Webhook is working!"));
app.get("/healthz", (_req, res) => {
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
  });
});

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
