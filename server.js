// ---- ลด log จาก TensorFlow (ต้องมาก่อน require tf) ----
process.env.TF_CPP_MIN_LOG_LEVEL = process.env.TF_CPP_MIN_LOG_LEVEL || "2";

require("dotenv").config();
const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

// ทำให้ sharp เสถียรบน Render
sharp.cache(true);
sharp.concurrency(1);

const app = express();
const PORT = process.env.PORT || 3000;
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;

// ===== Config (ค่าที่ปรับแล้ว) =====
const INPUT_SIZE = Number(process.env.INPUT_SIZE || 300);
const MODEL_INCLUDES_RESCALE = true; // โมเดลมี Rescaling(1./255) อยู่แล้ว

// 👇 ปรับ threshold ตามข้อเสนอ
const UNKNOWN_THRESHOLD = 0.4;   // เดิม 0.5
const MARGIN_THRESHOLD  = 0.07;  // เดิม 0.05
const ENTROPY_THRESHOLD = 1.6;   // คงเดิม
const SOFTMAX_TEMP      = 1.0;

// ===== 1) โหลด labels + model =====
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

let model = null;
let modelReady = false;
let modelType = "unknown";

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

// ===== 2) Helper: ตอบกลับ LINE =====
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
function softmaxTemp(arr, t=1) {
  const a = Array.from(arr, v => v / t);
  const m = Math.max(...a);
  const exps = a.map(v => Math.exp(v - m));
  const sum = exps.reduce((p, c) => p + c, 0);
  return exps.map(v => v / sum);
}

// ===== 3) Preprocess + Predict =====
async function classifyImage(imageBuffer, { debug = false } = {}) {
  if (!modelReady) throw new Error("Model not ready");

  const resized = await sharp(imageBuffer, { limitInputPixels: false })
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
    .toFormat("png")
    .toBuffer();

  let x = tf.node.decodeImage(resized, 3).toFloat().expandDims(0);
  if (!MODEL_INCLUDES_RESCALE) {
    x = x.div(255);
  }

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

  const raw = y.dataSync();
  let probs = Array.from(raw);
  const sum = probs.reduce((p, c) => p + c, 0);
  if (Math.abs(sum - 1) > 1e-3) probs = softmaxTemp(raw, SOFTMAX_TEMP);

  const { bestIdx, bestProb, secondProb } = top2(probs);
  const ent = entropy(probs);

  const unknown =
    (bestProb < UNKNOWN_THRESHOLD) ||
    (bestProb - secondProb < MARGIN_THRESHOLD) ||
    (ent > ENTROPY_THRESHOLD);

  const idx = unknown ? (labels.length - 1) : bestIdx;
  const label = labels[idx] || "ไม่สามารถจำแนกได้";
  const score = Number((bestProb * 100).toFixed(2));

  if (debug) {
    console.log("[DEBUG] probs:", probs.map(v => v.toFixed(4)));
    console.log(`[DEBUG] best=${bestProb.toFixed(4)} second=${secondProb.toFixed(4)} H=${ent.toFixed(4)}`);
  }

  tf.dispose([x, y]);
  return { label, score, appliedUnknown: unknown };
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
  thresholds: { UNKNOWN_THRESHOLD, MARGIN_THRESHOLD, ENTROPY_THRESHOLD }
}));

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
