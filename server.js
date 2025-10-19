// ---- ลด log TF (ต้องมาก่อน tf import) ----
process.env.TF_CPP_MIN_LOG_LEVEL = process.env.TF_CPP_MIN_LOG_LEVEL || "2";

require("dotenv").config();
const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

// --- sharp: ทำให้เสถียรกับไฟล์ใหญ่/EXIF ---
sharp.cache(true);
sharp.concurrency(1);

const app = express();
const PORT = process.env.PORT || 3000;
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;

// ====== CONFIG (มีค่า default ในตัว ไม่ต้องตั้ง ENV ก็ได้) ======
const INPUT_SIZE = Number(process.env.INPUT_SIZE || 300);

// โมเดลของเดียร์สร้างด้วย Rescaling(1./255) แล้ว
const MODEL_INCLUDES_RESCALE = true;

// Unknown policy
const UNKNOWN_THRESHOLD = Number(process.env.UNKNOWN_THRESHOLD || 0.55);
const MARGIN_THRESHOLD  = Number(process.env.MARGIN_THRESHOLD  || 0.08);
const ENTROPY_THRESHOLD = Number(process.env.ENTROPY_THRESHOLD || 1.60);

// ทำ prob ให้คมขึ้นนิดหน่อย (เพิ่มความมั่นใจ top-1 แบบไม่โอเวอร์)
const PROB_SHARPEN_GAMMA = Number(process.env.PROB_SHARPEN_GAMMA || 1.36);

// per-class calibration (ดัน Eczema/Shingles ให้เด่นขึ้น)
function parseDictEnv(text) {
  const out = {};
  if (!text) return out;
  String(text).split(",").forEach(pair => {
    const [k, v] = pair.split(":").map(s => s.trim());
    if (k && v && !Number.isNaN(Number(v))) out[k] = Number(v);
  });
  return out;
}
let CALIB_WEIGHTS = parseDictEnv(process.env.CALIB_WEIGHTS);
let PER_CLASS_THRESHOLDS = parseDictEnv(process.env.PER_CLASS_THRESHOLDS);

// defaults ถ้าไม่ตั้ง ENV
if (!Object.keys(CALIB_WEIGHTS).length) {
  CALIB_WEIGHTS = { Eczema: 1.55, Shingles: 1.45 };
}
if (!Object.keys(PER_CLASS_THRESHOLDS).length) {
  PER_CLASS_THRESHOLDS = { Eczema: 0.48, Shingles: 0.48 };
}

// ====== โหลด labels + model ======
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
let modelType = "unknown";

(async () => {
  try {
    // พยายามโหลด GraphModel ก่อน (จากที่เดียร์ export)
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
function softmax(arr) {
  const m = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / s);
}
function sharpenProbs(probs, gamma) {
  const raised = probs.map(p => Math.pow(Math.max(p, 1e-12), gamma));
  const s = raised.reduce((a, b) => a + b, 0);
  return raised.map(v => v / s);
}

// ===== Core: classify =====
async function classifyImage(imageBuffer, { debug = false } = {}) {
  if (!modelReady) throw new Error("Model not ready");

  // กันภาพแปลก ๆ: หมุนตาม EXIF, บังคับ RGB, ครอปกลาง, กันภาพใหญ่มาก
  let pre;
  try {
    pre = await sharp(imageBuffer, { limitInputPixels: false })
      .rotate() // ใช้ EXIF
      .ensureAlpha() // กันบางไฟล์ที่ไม่มี alpha channel
      .removeAlpha() // ให้ไป RGB 3 ช่อง
      .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
      .toFormat("png")
      .toBuffer();
  } catch (e) {
    throw new Error("IMAGE_PREPROCESS_FAIL: " + e.message);
  }

  // tensor
  let x = tf.node.decodeImage(pre, 3).toFloat().expandDims(0);
  if (!MODEL_INCLUDES_RESCALE) x = x.div(255);

  // predict รองรับทั้ง Graph/Layers
  let y = null;
  try {
    y = model.predict ? model.predict(x) : null;
    if (Array.isArray(y)) y = y[0];
    if (!y || typeof y.dataSync !== "function") {
      const feedName  = model.inputs?.[0]?.name;
      const fetchName = model.outputs?.[0]?.name;
      y = model.execute(feedName ? { [feedName]: x } : x, fetchName);
    }
  } catch (e) {
    tf.dispose(x);
    throw new Error("MODEL_EXEC_FAIL: " + e.message);
  }

  // แปลงผลเป็น prob
  let probs;
  try {
    const raw = Array.from(y.dataSync());
    const sum = raw.reduce((p, c) => p + c, 0);
    probs = (Math.abs(sum - 1) > 1e-3) ? softmax(raw) : raw;

    // calibration ตาม class
    const nameByIdx = labels;
    const weighted = probs.map((p, i) => {
      const name = nameByIdx[i] || "";
      const w = CALIB_WEIGHTS[name] || 1.0;
      return p * w;
    });
    const s = weighted.reduce((a, b) => a + b, 0);
    probs = weighted.map(v => v / (s || 1));

    // sharpen เพิ่มความคมของ top-1
    probs = sharpenProbs(probs, PROB_SHARPEN_GAMMA);
  } catch (e) {
    tf.dispose([x, y]);
    throw new Error("POSTPROCESS_FAIL: " + e.message);
  }

  const { bestIdx, bestProb, secondProb } = top2(probs);
  const bestName = labels[bestIdx] || "Unknown";
  const ent = entropy(probs);

  // threshold เฉพาะคลาส (ถ้าเซ็ตไว้)
  const perClassTh = PER_CLASS_THRESHOLDS[bestName];

  const isUnknown =
    (bestProb < (perClassTh ?? UNKNOWN_THRESHOLD)) ||
    (bestProb - secondProb < MARGIN_THRESHOLD) ||
    (ent > ENTROPY_THRESHOLD);

  const idx = isUnknown ? (labels.length - 1) : bestIdx;
  const label = labels[idx] || "Unknown";
  const score = Number((bestProb * 100).toFixed(2));

  if (debug) {
    const top3 = [...probs]
      .map((p, i) => ({ i, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 3)
      .map(o => `${labels[o.i]}:${(o.p * 100).toFixed(1)}%`)
      .join(", ");
    console.log(`[DEBUG] top3 = ${top3} | H=${ent.toFixed(3)} | margin=${(bestProb-secondProb).toFixed(3)}`);
  }

  tf.dispose([x, y]);
  return { label, score, appliedUnknown: isUnknown };
}

// ===== LINE Webhook =====
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
        // ดึงไฟล์จาก LINE
        let imgBuf;
        try {
          const r = await axios.get(
            `https://api-data.line.me/v2/bot/message/${event.message.id}/content`,
            {
              headers: { Authorization: `Bearer ${LINE_ACCESS_TOKEN}` },
              responseType: "arraybuffer",
              timeout: 30000,
              maxContentLength: Infinity,
              maxBodyLength: Infinity,
            }
          );
          imgBuf = Buffer.from(r.data);
        } catch (e) {
          console.error("Fetch image error:", e?.response?.status || e.message);
          await replyMessage(replyToken, "ดึงรูปจาก LINE ไม่สำเร็จ ลองส่งใหม่เป็น JPG/PNG ดูนะคะ");
          continue;
        }

        // จำแนก
        try {
          const { label, score, appliedUnknown } = await classifyImage(imgBuf, { debug: false });
          const extra = appliedUnknown ? " (จัดเป็น Unknown)" : "";
          await replyMessage(
            replyToken,
            `ผลการจำแนก: ${label}${extra}\nความเชื่อมั่นของคลาสสูงสุด ~${score}%`
          );
        } catch (e) {
          console.error("Classify error:", e.message);
          await replyMessage(
            replyToken,
            "ประมวลผลภาพไม่สำเร็จ กรุณาลองใหม่อีกครั้งค่ะ (ลองส่งเป็น JPG/PNG ขนาดไม่ใหญ่เกินไป)"
          );
        }
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

// ===== Health & Debug =====
app.get("/", (_req, res) => res.send("Webhook is working!"));
app.get("/healthz", (_req, res) => res.json({
  ok: true,
  modelReady,
  modelType,
  nLabels: labels.length,
  thresholds: {
    UNKNOWN_THRESHOLD,
    MARGIN_THRESHOLD,
    ENTROPY_THRESHOLD,
    PROB_SHARPEN_GAMMA,
    CALIB_WEIGHTS,
    PER_CLASS_THRESHOLDS
  }
}));

// ส่งภาพดิบทดสอบ (เช็ค error detail ใน log)
app.post("/debug/classify", express.raw({ type: "*/*", limit: "10mb" }), async (req, res) => {
  try {
    const out = await classifyImage(req.body, { debug: true });
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
