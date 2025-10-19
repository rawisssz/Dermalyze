// ลด log ของ TF (optional)
process.env.TF_CPP_MIN_LOG_LEVEL = process.env.TF_CPP_MIN_LOG_LEVEL || "2";

require("dotenv").config();
const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");
// const dialogflow = require("@google-cloud/dialogflow"); // ใช้ภายหลังค่อยเปิด

const app = express();
const PORT = process.env.PORT || 3000; // Render จะส่ง PORT ให้เอง
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;

app.use(bodyParser.json());

/* =========================
 * 1) โหลด labels + โมเดล
 * ========================= */
const MODEL_DIR = path.join(__dirname, "model");              // ต้องมี model.json + shard .bin
const MODEL_JSON = `file://${path.join(MODEL_DIR, "model.json")}`;
const LABELS_PATH = path.join(__dirname, "class_names.json"); // มี "Unknown" อยู่ท้ายสุด

// อ่าน labels
let labels = [];
try {
  const raw = fs.readFileSync(LABELS_PATH, "utf-8");
  labels = JSON.parse(raw);
  if (!Array.isArray(labels) || labels.length < 2) throw new Error("class_names.json ไม่ถูกต้อง");
  console.log("✅ Loaded labels:", labels);
} catch (e) {
  console.error("❌ โหลด class_names.json ไม่ได้:", e.message);
  // fallback กันล่ม (โปรดใส่ไฟล์จริงในโปรดักชัน)
  labels = ["ClassA", "ClassB", "Unknown"];
}

// พารามิเตอร์ inference
const INPUT_SIZE = 300;
const USE_UNKNOWN_THRESHOLD = true;
// ชั่วคราวตั้งต่ำหน่อยเพื่อเช็คว่าทายเป็นคลาสอื่นได้หรือไม่ (ค่อยปรับขึ้นภายหลัง)
const UNKNOWN_THRESHOLD = Number(process.env.UNKNOWN_THRESHOLD ?? 0.2);
const UNKNOWN_LABEL = labels[labels.length - 1] || "Unknown";

// ตัวแปรโมเดล
/** @type {tf.GraphModel | tf.LayersModel | null} */
let model = null;
let modelReady = false;
let modelType = "unknown"; // "graph" | "layers" | "unknown"

// โหลดโมเดล: พยายามโหลดแบบ GraphModel ก่อน ถ้าไม่ได้ค่อยลอง LayersModel
(async () => {
  try {
    model = await tf.loadGraphModel(MODEL_JSON);
    modelType = "graph";
    modelReady = true;
    console.log("✅ TFJS GraphModel loaded");
  } catch (gerr) {
    console.warn("ℹ️ loadGraphModel ล้มเหลว ลอง loadLayersModel ต่อ…", gerr?.message || gerr);
    try {
      model = await tf.loadLayersModel(MODEL_JSON);
      modelType = "layers";
      modelReady = true;
      console.log("✅ TFJS LayersModel loaded");
    } catch (lerr) {
      console.error("❌ Failed to load model both Graph & Layers:", lerr);
    }
  }
})();

/* =========================
 * 2) Helper: ส่งข้อความกลับ LINE
 * ========================= */
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

/* =========================
 * 3) ประมวลผลภาพ + จำแนก
 * ========================= */
async function classifyImage(imageBuffer) {
  if (!model || !modelReady) throw new Error("Model is not loaded yet");

  // 1) preprocess ด้วย sharp
  const resized = await sharp(imageBuffer, { limitInputPixels: false }) // ปิด limit ป้องกันรูปใหญ่
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
    .toFormat("png")
    .toBuffer();

  // 2) สร้างเทนเซอร์ [1,H,W,3] และ normalize /255
  const x = tf.node.decodeImage(resized, 3).toFloat().div(255).expandDims(0);

  // 3) forward (รองรับทั้งสองชนิด)
  let y; // Tensor
  if (modelType === "layers" && typeof (/** @type any */(model)).predict === "function") {
    y = (/** @type tf.LayersModel */(model)).predict(x);
  } else {
    // GraphModel: ใช้ executeAsync พร้อม map input/output name
    const g = /** @type tf.GraphModel */(model);
    const inName = g.inputs?.[0]?.name;      // เช่น 'serving_default_input_1'
    const outName = g.outputs?.[0]?.name;    // เช่น 'StatefulPartitionedCall:0'
    if (!inName || !outName) throw new Error("ไม่พบชื่อ input/output ของ GraphModel");
    const out = await g.executeAsync({ [inName]: x }, outName);
    y = Array.isArray(out) ? out[0] : out;
  }

  // 4) ให้ผลลัพธ์เป็นความน่าจะเป็นเสมอ (softmax หากยังเป็น logits)
  let probsT = /** @type tf.Tensor */(y);
  // ถ้าผลรวมโพรบาไม่ใกล้ 1 ให้ softmax เอง
  const sum = (await probsT.sum().data())[0];
  if (!Number.isFinite(sum) || Math.abs(sum - 1) > 1e-3) {
    probsT = tf.softmax(probsT);
  }
  const probs = await probsT.data();

  // 5) เลือกคลาส
  let maxProb = -1, maxIdx = -1;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > maxProb) { maxProb = probs[i]; maxIdx = i; }
  }

  // threshold → Unknown (label ท้ายสุด)
  let finalIdx = maxIdx;
  if (USE_UNKNOWN_THRESHOLD && maxProb < UNKNOWN_THRESHOLD) {
    finalIdx = labels.length - 1;
  }

  const label = labels[finalIdx] || "ไม่สามารถจำแนกได้";
  const score = Number((probs[maxIdx] * 100).toFixed(2));

  tf.dispose([x, y, probsT]); // เก็บหน่วยความจำ

  return { label, score, maxProb, appliedUnknown: finalIdx !== maxIdx };
}

/* =========================
 * 4) LINE Webhook
 * ========================= */
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
        // ดึงรูปจาก LINE
        const imageId = event.message.id;
        const imgResp = await axios.get(
          `https://api-data.line.me/v2/bot/message/${imageId}/content`,
          { headers: { Authorization: `Bearer ${LINE_ACCESS_TOKEN}` }, responseType: "arraybuffer", timeout: 20000 }
        );

        // จำแนก
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

/* =========================
 * 5) Health / Debug
 * ========================= */
app.get("/", (_req, res) => res.send("Webhook is working!"));

app.get("/healthz", (_req, res) => res.json({
  ok: true, modelReady, modelType,
  labels: labels.length, threshold: UNKNOWN_THRESHOLD
}));

// ใช้ตรวจชื่อ input/output ของโมเดล (มีประโยชน์มากเวลาเป็น GraphModel)
app.get("/debug", (_req, res) => {
  res.json({
    modelReady,
    modelType,
    inputs: (/** @type any */(model))?.inputs?.map(i => i.name),
    outputs: (/** @type any */(model))?.outputs?.map(o => o.name),
  });
});

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));