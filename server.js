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

const INPUT_SIZE = Number(process.env.INPUT_SIZE || 300);
// ทำให้ไม่ฟาด Unknown ง่ายเกินไป
const UNKNOWN_THRESHOLD = Number(process.env.UNKNOWN_THRESHOLD || 0.35);
const MARGIN_THRESHOLD  = Number(process.env.MARGIN_THRESHOLD  || 0.05);

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

// -------- utils --------
function softmax(arr) {
  const m = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - m));
  const s = exps.reduce((p, c) => p + c, 0);
  return exps.map(v => v / s);
}
function top2(probArray) {
  let bI = -1, bP = -1, sI = -1, sP = -1;
  for (let i = 0; i < probArray.length; i++) {
    const p = probArray[i];
    if (p > bP) { sI = bI; sP = bP; bI = i; bP = p; }
    else if (p > sP) { sI = i; sP = p; }
  }
  return { bestIdx: bI, bestProb: bP, secondIdx: sI, secondProb: sP };
}

// -------- core classify --------
async function classifyImage(imageBuffer, { debug = false } = {}) {
  if (!model || !modelReady) throw new Error("Model is not loaded yet");

  const resized = await sharp(imageBuffer, { limitInputPixels: false })
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
    .toFormat("png")
    .toBuffer();

  const x = tf.node.decodeImage(resized, 3).toFloat().div(255).expandDims(0);

  let out = model.predict ? model.predict(x) : null;
  if (Array.isArray(out)) out = out[0];

  if (!out || typeof out.dataSync !== "function") {
    try {
      const feedName  = model.inputs?.[0]?.name;
      const fetchName = model.outputs?.[0]?.name;
      out = model.execute(feedName ? { [feedName]: x } : { x }, fetchName);
    } catch (e) {
      tf.dispose(x);
      throw e;
    }
  }

  const raw = Array.from(out.dataSync());
  tf.dispose([x, out]);

  const rawMin = Math.min(...raw);
  const rawMax = Math.max(...raw);
  const rawSum = raw.reduce((p, c) => p + c, 0);

  // ถ้าดูเหมือน prob แล้ว (อยู่ใน [0,1] และผลรวมใกล้ 1) ใช้ raw เลย
  const looksLikeProb = rawMin >= -1e-6 && rawMax <= 1 + 1e-6 && Math.abs(rawSum - 1) < 1e-2;
  const probs = looksLikeProb ? raw : softmax(raw);

  const { bestIdx, bestProb, secondProb } = top2(probs);
  const unknown = (bestProb < UNKNOWN_THRESHOLD) || (bestProb - secondProb < MARGIN_THRESHOLD);

  if (debug) {
    console.log(`[DEBUG] sum(raw)=${rawSum.toFixed(4)} min=${rawMin.toFixed(4)} max=${rawMax.toFixed(4)} ` +
                `mode=${looksLikeProb ? "as-is" : "softmax"}`);
    console.log("[DEBUG] probs:", probs.map(v => Number(v.toFixed(4))));
    console.log("[DEBUG] top1:", bestIdx, bestProb.toFixed(4), " top2:", secondProb.toFixed(4));
  }

  const idx = unknown ? (labels.length - 1) : bestIdx;
  return {
    label: labels[idx] || "ไม่สามารถจำแนกได้",
    score: Number((bestProb * 100).toFixed(2)),
    appliedUnknown: unknown
  };
}

// -------- webhook --------
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
        const { label, score, appliedUnknown } = await classifyImage(imgResp.data, { debug: false });
        await replyMessage(
          replyToken,
          `ผลการจำแนก: ${label}${appliedUnknown ? " (จัดเป็น Unknown)" : ""}\nความเชื่อมั่นของคลาสสูงสุด ~${score}%`
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

// -------- health & debug --------
app.get("/", (_req, res) => res.send("Webhook is working!"));
app.get("/healthz", (_req, res) =>
  res.json({ ok: true, modelReady, modelType, nLabels: labels.length,
             thresholds: { UNKNOWN_THRESHOLD, MARGIN_THRESHOLD } })
);

// debug endpoint: ส่งไฟล์ภาพ (body เป็นไบนารี) เพื่อดู log
app.post("/debug/classify", express.raw({ type: "*/*", limit: "10mb" }), async (req, res) => {
  try { res.json(await classifyImage(req.body, { debug: true })); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
