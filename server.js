require("dotenv").config();
const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");
// const dialogflow = require("@google-cloud/dialogflow"); // จะใช้ทีหลังค่อยปลดคอมเมนต์

const app = express();
const PORT = process.env.PORT || 3000;
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;

app.use(bodyParser.json());

/* =========================
 * 1) โหลด labels + โมเดล
 * ========================= */
const MODEL_DIR = path.join(__dirname, "model");           // ต้องมี model.json + shard .bin
const LABELS_PATH = path.join(__dirname, "class_names.json"); // มาจากสคริปต์เทรน (รวม "Unknown" ท้ายสุด)

let labels = [];
try {
  const raw = fs.readFileSync(LABELS_PATH, "utf-8");
  labels = JSON.parse(raw);
  if (!Array.isArray(labels) || labels.length < 2) {
    throw new Error("class_names.json ไม่ถูกต้อง");
  }
  console.log("✅ Loaded labels:", labels);
} catch (e) {
  console.error("❌ โหลด class_names.json ไม่ได้:", e.message);
  // fallback (กำหนดคร่าว ๆ เพื่อไม่ให้ล่ม—ควรมีไฟล์จริงในโปรดักชัน)
  labels = ["ClassA", "ClassB", "Unknown"];
}

// อิงสคริปต์เทรน: IMG_SIZE = (300, 300) และใช้ Rescaling(1./255)
const INPUT_SIZE = 300;

// ถ้าคุณอยากตัดเข้า Unknown ด้วย threshold:
// - Unknown อยู่ท้ายสุดของ labels (ตามตอนเซฟ)
// - ถ้า maxProb < THRESHOLD → บังคับเป็น "Unknown"
const USE_UNKNOWN_THRESHOLD = true;
const UNKNOWN_THRESHOLD = 0.50; // ปรับได้ เช่น 0.5 ตามโน้ตในสคริปต์เทรน
const UNKNOWN_LABEL = labels[labels.length - 1] || "Unknown";

let model;
(async () => {
  try {
    model = await tf.loadLayersModel(`file://${path.join(MODEL_DIR, "model.json")}`);
    console.log("✅ TFJS model loaded");
  } catch (err) {
    console.error("❌ Failed to load model:", err);
  }
})();

/* =========================
 * 2) Helper: ตอบกลับ LINE
 * ========================= */
async function replyMessage(replyToken, text) {
  try {
    await axios.post(
      "https://api.line.me/v2/bot/message/reply",
      {
        replyToken,
        messages: [{ type: "text", text }],
      },
      {
        headers: { Authorization: `Bearer ${LINE_ACCESS_TOKEN}` },
        timeout: 15000,
      }
    );
  } catch (e) {
    console.error("Reply error:", e?.response?.data || e.message);
  }
}

/* =========================
 * 3) Preprocess + Predict
 * ========================= */
// หมายเหตุ: ในสคริปต์เทรนคุณใช้ Rescaling(1./255) และ include_preprocessing=False
// ดังนั้นฝั่ง inference ต้องทำ normalize เองให้ตรง: /255.0
async function classifyImage(imageBuffer) {
  if (!model) throw new Error("Model is not loaded yet");

  // resize → PNG → tensor → normalize → [1,H,W,3]
  const resized = await sharp(imageBuffer)
    .resize(INPUT_SIZE, INPUT_SIZE)
    .toFormat("png")
    .toBuffer();

  const tensor = tf.node
    .decodeImage(resized, 3)  // [H,W,3]
    .toFloat()
    .div(255.0)               // ตรงกับ Rescaling(1./255)
    .expandDims(0);           // [1,H,W,3]

  const logits = model.predict(tensor);
  const probs = await logits.data(); // Float32Array ความยาว = labels.length
  const numClasses = labels.length;

  // หา top-1
  let maxProb = -1;
  let maxIdx = -1;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > maxProb) {
      maxProb = probs[i];
      maxIdx = i;
    }
  }

  // บังคับ Unknown ด้วย threshold (ออปชัน)
  let finalIdx = maxIdx;
  if (USE_UNKNOWN_THRESHOLD && maxProb < UNKNOWN_THRESHOLD) {
    finalIdx = numClasses - 1; // ชี้ไป Unknown (ท้ายสุด)
  }

  const label = labels[finalIdx] || "ไม่สามารถจำแนกได้";
  const score = Number((probs[maxIdx] * 100).toFixed(2)); // รายงานความเชื่อมั่นของ top-1

  // เก็บหน่วยความจำ
  tf.dispose([tensor, logits]);

  return { label, score, maxIdx, maxProb, appliedUnknown: finalIdx !== maxIdx };
}

/* =========================
 * 4) LINE Webhook
 * ========================= */
app.post("/webhook", async (req, res) => {
  const events = req.body?.events || [];
  for (const event of events) {
    const replyToken = event.replyToken;

    try {
      // รับ "รูปภาพ"
      if (event.type === "message" && event.message.type === "image") {
        const imageId = event.message.id;

        // 1) ดึงภาพจาก LINE
        const imgResp = await axios.get(
          `https://api-data.line.me/v2/bot/message/${imageId}/content`,
          {
            headers: { Authorization: `Bearer ${LINE_ACCESS_TOKEN}` },
            responseType: "arraybuffer",
            timeout: 20000,
          }
        );

        // 2) จำแนก
        const { label, score, appliedUnknown } = await classifyImage(imgResp.data);

        // 3) ตอบกลับ (ตอนนี้ยังไม่เรียก Dialogflow — คอมเมนต์ไว้ก่อน)
        const extra = appliedUnknown ? " (จัดเป็น Unknown โดย threshold)" : "";
        await replyMessage(
          replyToken,
          `ผลการจำแนก: ${label}${extra}\nความเชื่อมั่นของคลาสสูงสุด ~${score}%`
        );

        // ====== (เตรียมไว้) เรียก Dialogflow หลังจำแนก ======
        // const userId = event?.source?.userId || "anon";
        // const sessionClient = new dialogflow.SessionsClient();
        // const sessionPath = sessionClient.projectAgentSessionPath(process.env.DIALOGFLOW_PROJECT_ID, userId);
        // const ask = `ข้อมูลโรค ${label}`; // Intent: DiseaseInfo
        // const request = {
        //   session: sessionPath,
        //   queryInput: { text: { text: ask, languageCode: "th" } },
        // };
        // const responses = await sessionClient.detectIntent(request);
        // const dfReply = responses?.[0]?.queryResult?.fulfillmentText || "";
        // if (dfReply) await replyMessage(replyToken, dfReply);

      } else if (event.type === "message" && event.message.type === "text") {
        // ข้อความทั่วไป (ยังไม่เรียก Dialogflow)
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
 * 5) Health check
 * ========================= */
app.get("/", (_req, res) => res.send("Webhook is working!"));

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
