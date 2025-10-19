// ลด log TF
process.env.TF_CPP_MIN_LOG_LEVEL = process.env.TF_CPP_MIN_LOG_LEVEL || "2";

require("dotenv").config();
const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000; // Render จะกำหนดให้เอง
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;
const DEBUG_PRED = process.env.DEBUG_PRED === "1";

app.use(bodyParser.json());

/* ========== 1) โหลด labels + โมเดล ========== */
const MODEL_DIR = path.join(__dirname, "model");
const MODEL_JSON = `file://${path.join(MODEL_DIR, "model.json")}`;
const LABELS_PATH = path.join(__dirname, "class_names.json");

let labels = [];
try {
  labels = JSON.parse(fs.readFileSync(LABELS_PATH, "utf-8"));
  if (!Array.isArray(labels) || labels.length < 2) throw new Error("labels invalid");
  console.log("✅ Loaded labels:", labels);
} catch (e) {
  console.error("❌ โหลด class_names.json ไม่ได้:", e.message);
  labels = ["ClassA", "ClassB", "Unknown"];
}

const NUM_CLASSES = labels.length;
const INPUT_SIZE = 300;
const USE_UNKNOWN_THRESHOLD = true;
const UNKNOWN_THRESHOLD = Number(process.env.UNKNOWN_THRESHOLD ?? 0.0); // เริ่ม 0 ก่อนเพื่อทดสอบ

/** @type {tf.GraphModel | tf.LayersModel | null} */
let model = null;
let modelReady = false;
let modelType = "unknown"; // "graph" | "layers"

(async () => {
  try {
    model = await tf.loadGraphModel(MODEL_JSON);
    modelType = "graph";
    modelReady = true;
    console.log("✅ TFJS GraphModel loaded");
  } catch (gerr) {
    console.warn("ℹ️ loadGraphModel ล้มเหลว ลอง Layers ต่อ…", gerr?.message || gerr);
    try {
      model = await tf.loadLayersModel(MODEL_JSON);
      modelType = "layers";
      modelReady = true;
      console.log("✅ TFJS LayersModel loaded");
    } catch (lerr) {
      console.error("❌ โหลดโมเดลไม่สำเร็จ (Graph & Layers):", lerr);
    }
  }
})();

/* ========== 2) Helper LINE ========== */
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

/* ========== 3) เลือก output ที่ถูก (GraphModel) ========== */
const lastDimEquals = (t, n) => t?.shape?.length >= 1 && t.shape[t.shape.length - 1] === n;
/** @param {tf.Tensor|tf.Tensor[]} out */
function pickBestLogits(out, numClasses) {
  const arr = Array.isArray(out) ? out : [out];
  const candidates = arr.filter(t => lastDimEquals(t, numClasses));
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => a.shape.length - b.shape.length); // เอา rank ต่ำสุด
  return candidates[0];
}

/* ========== 4) Inference ========== */
// สร้างภาพเป็นเทนเซอร์ [1,H,W,3] แบบต่าง ๆ
async function makeInputTensors(imageBuffer) {
  const resized = await sharp(imageBuffer, { limitInputPixels: false })
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
    .toFormat("png")
    .toBuffer();

  const base = tf.node.decodeImage(resized, 3);     // uint8 [H,W,3] 0..255
  const xRaw255 = base.toFloat().expandDims(0);     // [1,H,W,3] 0..255
  const xDiv255 = base.toFloat().div(255).expandDims(0); // [1,H,W,3] 0..1
  // ImageNet แบบง่าย (mean/std ของ tf.keras.applications)
  const mean = tf.tensor1d([0.485, 0.456, 0.406]);
  const std = tf.tensor1d([0.229, 0.224, 0.225]);
  const xImagenet = base.toFloat().div(255).sub(mean).div(std).expandDims(0);

  base.dispose(); mean.dispose(); std.dispose();
  return { xRaw255, xDiv255, xImagenet };
}

// รันโมเดล 1 ครั้ง ให้คืน probs และ maxProb
async function forwardOnce(x) {
  let logits;
  if (modelType === "layers") {
    let y = /** @type {tf.LayersModel} */(model).predict(x);
    logits = Array.isArray(y) ? y[0] : y;
  } else {
    const g = /** @type {tf.GraphModel} */(model);
    const inName = g.inputs?.[0]?.name;
    if (!inName) throw new Error("GraphModel: ไม่พบชื่อ input");
    const rawOut = await g.executeAsync({ [inName]: x });
    let chosen = pickBestLogits(rawOut, NUM_CLASSES) || (Array.isArray(rawOut) ? rawOut[0] : rawOut);
    logits = chosen;
  }
  if (logits.shape.length > 1) logits = logits.squeeze(); // → [C]
  const probsT = tf.softmax(logits);
  const probs = await probsT.data();
  const maxProb = Math.max(...probs);
  tf.dispose([logits, probsT]);
  return { probs, maxProb };
}

async function classifyImage(imageBuffer) {
  if (!model || !modelReady) throw new Error("Model not ready");

  const { xRaw255, xDiv255, xImagenet } = await makeInputTensors(imageBuffer);

  // ลองทั้งสามโหมด แล้วเลือกตัวที่ maxProb สูงสุด
  const tries = [];
  tries.push({ mode: "raw255", ...(await forwardOnce(xRaw255)) });
  tries.push({ mode: "div255", ...(await forwardOnce(xDiv255)) });
  tries.push({ mode: "imagenet", ...(await forwardOnce(xImagenet)) });

  // ทำลายอินพุต
  tf.dispose([xRaw255, xDiv255, xImagenet]);

  tries.sort((a, b) => b.maxProb - a.maxProb);
  const best = tries[0];
  const probs = best.probs;

  // หา index สูงสุด
  let maxIdx = 0, maxProb = probs[0];
  for (let i = 1; i < probs.length; i++) if (probs[i] > maxProb) { maxProb = probs[i]; maxIdx = i; }

  // threshold → Unknown
  let finalIdx = maxIdx;
  if (USE_UNKNOWN_THRESHOLD && maxProb < UNKNOWN_THRESHOLD) finalIdx = NUM_CLASSES - 1;

  if (DEBUG_PRED) {
    const top5 = [...probs].map((p, i) => ({ i, lbl: labels[i], p:+p.toFixed(4) }))
      .sort((a,b)=>b.p-a.p).slice(0,5);
    console.log(`🔎 mode=${best.mode} maxProb=${maxProb.toFixed(4)} top5=`, top5);
  }

  const label = labels[finalIdx] || "ไม่สามารถจำแนกได้";
  const score = Number((probs[maxIdx] * 100).toFixed(2));
  return { label, score, maxProb, appliedUnknown: finalIdx !== maxIdx, mode: best.mode };
}

/* ========== 5) LINE Webhook ========== */
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
          { headers: { Authorization: `Bearer ${LINE_ACCESS_TOKEN}` }, responseType: "arraybuffer", timeout: 20000 }
        );

        const { label, score, appliedUnknown } = await classifyImage(imgResp.data);
        const extra = appliedUnknown ? " (จัดเป็น Unknown โดย threshold)" : "";
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

/* ========== 6) Health / Debug ========== */
app.get("/", (_req, res) => res.send("Webhook is working!"));
app.get("/healthz", (_req, res) => res.json({
  ok: true, modelReady, modelType, labels: labels.length, threshold: UNKNOWN_THRESHOLD
}));
app.get("/debug", (_req, res) => {
  res.json({
    modelReady,
    modelType,
    inputs: (/** @type any */(model))?.inputs?.map(i => i.name),
    outputs: (/** @type any */(model))?.outputs?.map(o => o.name),
  });
});

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));