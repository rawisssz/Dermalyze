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

// ปรับพฤติกรรม sharp ให้นิ่งขึ้นบนโฮสติ้ง
sharp.cache(true);
sharp.concurrency(1);

const app = express();
const PORT = process.env.PORT || 3000;            // Render จะส่ง PORT มาให้เอง
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;

// ---- ค่า config ปรับได้ผ่าน env ----
const INPUT_SIZE = Number(process.env.INPUT_SIZE || 300);
const UNKNOWN_THRESHOLD = Number(process.env.UNKNOWN_THRESHOLD || 0.50); // สงสัย not-sure → Unknown
const MARGIN_THRESHOLD  = Number(process.env.MARGIN_THRESHOLD  || 0.05); // ต่างจากอันดับ 2 น้อยเกินไป → Unknown
const ENTROPY_THRESHOLD = Number(process.env.ENTROPY_THRESHOLD || 1.60); // กระจายมาก → Unknown
const SOFTMAX_TEMP      = Number(process.env.SOFTMAX_TEMP      || 1.0);  // >1 ทำให้โปรบแบนลง

// ======================================================
// 1) โหลด labels + model
// ======================================================
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
// 2) ฟังก์ชันตอบ LINE
// ======================================================
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

// ======================================================
/** Utils */
function softmaxTemp(arr, temp = 1.0) {
  const a = Array.from(arr, v => v / temp);
  const m = Math.max(...a);
  const exps = a.map(v => Math.exp(v - m));
  const sum = exps.reduce((p, c) => p + c, 0);
  return exps.map(v => v / sum);
}
function entropy(probArray) {
  let h = 0;
  for (const p of probArray) if (p > 0) h -= p * Math.log(p);
  return h;
}
function top2(probArray) {
  let best = [-1, -1], second = [-1, -1];
  for (let i = 0; i < probArray.length; i++) {
    const p = probArray[i];
    if (p > best[1]) { second = best; best = [i, p]; }
    else if (p > second[1]) { second = [i, p]; }
  }
  return { bestIdx: best[0], bestProb: best[1], secondIdx: second[0], secondProb: second[1] };
}

// ======================================================
// 3) Preprocess + Predict + Unknown policy (+ nLogits debug)
// ======================================================
async function classifyImage(imageBuffer, { debug = false } = {}) {
  if (!model || !modelReady) throw new Error("Model is not loaded yet");

  // resize → PNG → tensor [1,H,W,3] → normalize 0..1
  const resized = await sharp(imageBuffer, { limitInputPixels: false })
    .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
    .toFormat("png")
    .toBuffer();

  const x = tf.node.decodeImage(resized, 3).toFloat().div(255).expandDims(0);

  let y = model.predict ? model.predict(x) : null;
  if (Array.isArray(y)) y = y[0];

  // GraphModel fallback
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
  // ถ้าไม่ใช่ probs (เช่น logits) → ทำ softmax เอง
  const sum = raw.reduce((p, c) => p + c, 0);
  const probs = (Math.abs(sum - 1) > 1e-3 || raw.some(v => v < 0) || raw.some(v => v > 1))
    ? softmaxTemp(raw, SOFTMAX_TEMP)
    : Array.from(raw);

  const nLogits = probs.length;
  if (nLogits !== labels.length) {
    console.warn(`⚠️ MISMATCH: model outputs ${nLogits} classes but labels has ${labels.length}. Check class_names.json vs training classes.`);
  }

  const { bestIdx, bestProb, secondProb } = top2(probs);
  const ent = entropy(probs);

  const unknown =
    (bestProb < UNKNOWN_THRESHOLD) ||
    (bestProb - secondProb < MARGIN_THRESHOLD) ||
    (ent > ENTROPY_THRESHOLD);

  // ใช้ label เท่าที่มีจริงเพื่อกันพลาดระหว่างแก้โมเดล/labels
  const activeLabels = labels.slice(0, nLogits);
  const finalIdx = unknown ? -1 : bestIdx;
  const label = finalIdx === -1
    ? (labels[labels.length - 1] || "Unknown")
    : (activeLabels[finalIdx] || "ไม่สามารถจำแนกได้");
  const score = Number((bestProb * 100).toFixed(2));

  if (debug) {
    console.log("[DEBUG] probs:", probs.map(v => Number(v.toFixed(4))));
    console.log(`[DEBUG] nLogits=${nLogits}, nLabels=${labels.length}, best=${bestProb.toFixed(4)}, second=${secondProb.toFixed(4)}, H=${ent.toFixed(4)}`);
  }

  tf.dispose([x, y]);
  return { label, score, appliedUnknown: unknown, nLogits };
}

// ======================================================
// 4) LINE Webhook
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
        const imgResp = await axios.get(
          `https://api-data.line.me/v2/bot/message/${imageId}/content`,
          {
            headers: { Authorization: `Bearer ${LINE_ACCESS_TOKEN}` },
            responseType: "arraybuffer",
            timeout: 20000,
          }
        );

        const { label, score, appliedUnknown, nLogits } = await classifyImage(imgResp.data);
        const extra = appliedUnknown ? " (จัดเป็น Unknown)" : "";
        await replyMessage(
          replyToken,
          `ผลการจำแนก: ${label}${extra}\nความเชื่อมั่นของคลาสสูงสุด ~${score}%`
          // ถ้าอยากดู nLogits ที่ไลน์ให้แสดงด้วย ให้ต่อ \n[nLogits:${nLogits}]
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

// ======================================================
// 5) Health & Debug endpoint
// ======================================================
app.get("/", (_req, res) => res.send("Webhook is working!"));
app.get("/healthz", (_req, res) => res.json({
  ok: true, modelReady, modelType,
  nLabels: labels.length,
  thresholds: { UNKNOWN_THRESHOLD, MARGIN_THRESHOLD, ENTROPY_THRESHOLD, SOFTMAX_TEMP }
}));

// ส่งไฟล์ไบนารี (body เป็นรูป) เพื่อดีบักได้ — ใช้แค่ทดสอบ
app.post("/debug/classify", express.raw({ type: "*/*", limit: "10mb" }), async (req, res) => {
  try {
    const out = await classifyImage(req.body, { debug: true });
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
