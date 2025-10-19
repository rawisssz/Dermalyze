// ลด log TensorFlow (ต้องตั้งก่อน require tf)
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
const PORT = process.env.PORT || 3000; // Render จะส่ง PORT มาให้
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;

app.use(bodyParser.json());

/* ========= 1) โหลด labels + โมเดล ========= */
const MODEL_DIR = path.join(__dirname, "model");
const MODEL_PATH = `file://${path.join(MODEL_DIR, "model.json")}`;
const LABELS_PATH = path.join(__dirname, "class_names.json");

let labels = [];
try {
  labels = JSON.parse(fs.readFileSync(LABELS_PATH, "utf-8"));
  if (!Array.isArray(labels) || labels.length < 2) throw new Error("รูปแบบ labels ไม่ถูก");
  console.log("✅ Labels:", labels);
} catch (e) {
  console.error("❌ โหลด labels ไม่ได้:", e.message);
  labels = ["ClassA", "ClassB", "Unknown"];
}

const INPUT_SIZE = 300;
const USE_UNKNOWN_THRESHOLD = true;
const UNKNOWN_THRESHOLD = 0.5;

let model = null;
let modelReady = false;
let modelType = "unknown"; // "layers" | "graph"

(async () => {
  try {
    try {
      model = await tf.loadLayersModel(MODEL_PATH);
      modelType = "layers";
      modelReady = true;
      console.log("✅ Loaded TFJS LayersModel");
    } catch (e1) {
      console.warn("ℹ️ Not a LayersModel, trying GraphModel…", e1?.message);
      model = await tf.loadGraphModel(MODEL_PATH);
      modelType = "graph";
      modelReady = true;
      console.log("✅ Loaded TFJS GraphModel");
    }
  } catch (err) {
    console.error("❌ Failed to load model:", err);
  }
})();

/* ========= 2) Helper: ส่งข้อความกลับ LINE ========= */
async function replyMessage(replyToken, text) {
  try {
    await axios.post(
      "https://api.line.me/v2/bot/message/reply",
      { replyToken, messages: [{ type: "text", text }] },
      {
        headers: { Authorization: `Bearer ${LINE_ACCESS_TOKEN}` },
        timeout: 15000,
      }
    );
  } catch (e) {
    console.error("Reply error:", e?.response?.data || e.message);
  }
}

/* ========= 3) ประมวลผลภาพ + จำแนก ========= */
async function classifyImage(imageBuffer) {
  if (!model || !modelReady) throw new Error("Model is not loaded yet");

  const resized = await sharp(imageBuffer, { limitInputPixels: false })
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
    .toFormat("png")
    .toBuffer();

  const x = tf.node.decodeImage(resized, 3).toFloat().div(255).expandDims(0);

  let out = model.predict(x);
  if (Array.isArray(out)) out = out[0];

  if (!out || typeof out.dataSync !== "function") {
    const outputs = model.outputs ? model.outputs.map(o => o.name) : undefined;
    out = model.execute ? model.execute({ [model.inputs[0].name]: x }, outputs?.[0]) : out;
  }

  const probs = out.dataSync();
  let maxProb = -1, maxIdx = -1;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > maxProb) { maxProb = probs[i]; maxIdx = i; }
  }

  let finalIdx = maxIdx;
  if (USE_UNKNOWN_THRESHOLD && maxProb < UNKNOWN_THRESHOLD) {
    finalIdx = labels.length - 1; // สมมติ index ท้ายคือ Unknown
  }

  const label = labels[finalIdx] || "ไม่สามารถจำแนกได้";
  const score = Number((probs[maxIdx] * 100).toFixed(2));

  tf.dispose([x, out]);
  return { label, score, maxProb, appliedUnknown: finalIdx !== maxIdx };
}

/* ========= 4) LINE Webhook ========= */
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

/* ========= 5) Health Check ========= */
app.get("/", (_req, res) => res.send("Webhook is working!"));
app.get("/healthz", (_req, res) =>
  res.json({ ok: true, modelReady, modelType, labels: labels.length })
);

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));