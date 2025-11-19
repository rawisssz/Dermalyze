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
const dialogflow = require("@google-cloud/dialogflow");
const { google } = require("googleapis");

// --- sharp: ทำให้เสถียรกับไฟล์ใหญ่/EXIF ---
sharp.cache(true);
sharp.concurrency(1);

app.use("/static", express.static(path.join(__dirname, "public")));
const app = express();
const PORT = process.env.PORT || 3000;
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;

// ====== CONFIG (มีค่า default ในตัว ไม่ต้องตั้ง ENV ก็ได้) ======
const INPUT_SIZE = Number(process.env.INPUT_SIZE || 300);

// โมเดลของเดียร์สร้างด้วย Rescaling(1./255) แล้ว
const MODEL_INCLUDES_RESCALE = true;

// Unknown policy
const UNKNOWN_THRESHOLD = Number(process.env.UNKNOWN_THRESHOLD || 0.55);
const MARGIN_THRESHOLD = Number(process.env.MARGIN_THRESHOLD || 0.08);
const ENTROPY_THRESHOLD = Number(process.env.ENTROPY_THRESHOLD || 1.60);

// ทำ prob ให้คมขึ้นนิดหน่อย (เพิ่มความมั่นใจ top-1 แบบไม่โอเวอร์)
const PROB_SHARPEN_GAMMA = Number(process.env.PROB_SHARPEN_GAMMA || 1.36);

// ====== ENV keywords จาก Rich menu ======
const START_QUIZ_KEYWORD =
  process.env.START_QUIZ_KEYWORD || "เริ่มทำแบบประเมินผื่นผิวหนัง";
const USER_GUIDE_KEYWORD =
  process.env.USER_GUIDE_KEYWORD || "คู่มือการใช้งาน";
const OUTBREAK_KEYWORD =
  process.env.OUTBREAK_KEYWORD || "โรคผิวหนังที่กำลังระบาด";

// ===== URL รูปที่ใช้ในเมนู C / E (ใช้ชื่อเดียวกับ ENV บน Render) =====
const USER_GUIDE_IMAGE_URL =
  process.env.USER_GUIDE_IMAGE_URL ||
  "https://github.com/rawisssz/Dermalyze/blob/main/public/images/user_guide.png?raw=true";
const OUTBREAK_IMAGE_URL =
  process.env.OUTBREAK_IMAGE_URL ||
  "https://github.com/rawisssz/Dermalyze/blob/main/public/images/outbreak.jpg?raw=true";

// per-class calibration (ดัน Eczema/Shingles ให้เด่นขึ้น)
function parseDictEnv(text) {
  const out = {};
  if (!text) return out;
  String(text)
    .split(",")
    .forEach((pair) => {
      const [k, v] = pair.split(":").map((s) => s.trim());
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

// ====== DISCLAIMER (ข้อความเตือน แปะท้ายทุกคำตอบที่มี info/care) ======
const DISCLAIMER = `⚠️ ข้อจำกัดสำคัญ
ข้อมูลนี้เป็นเพียงการประเมิน/ให้ข้อมูลเบื้องต้นจากระบบ ไม่สามารถใช้แทนการวินิจฉัยหรือคำแนะนำจากแพทย์ได้
หากมีอาการรุนแรง ผื่นลามเร็ว เจ็บปวดมาก ไข้สูง ผื่นใกล้ตา/ใบหน้า หรือกังวลใจเกี่ยวกับอาการของตนเอง
ควรไปพบแพทย์หรือพบแพทย์ผิวหนังเพื่อรับการประเมินอย่างเหมาะสมทุกครั้งนะคะ`;

// ====== Disease Info/Care mapping (ไม่มี DISCLAIMER ฝังอยู่แล้ว) ======
const diseaseInfo = {
  Acne: {
    info: `"สิว" (Acne)
- สิวเกิดจากต่อมไขมันทำงานมากขึ้น + การอุดตันของรูขุมขน + เชื้อ P.acnes
- มักขึ้นที่ใบหน้า หน้าอก หลัง
- มีทั้งสิวอุดตัน (หัวขาว/หัวดำ) และสิวอักเสบ (แดง หนอง ก้อนลึก)`,
    care: `การดูแลเบื้องต้นสำหรับ "สิว"
1) ล้างหน้าเบา ๆ วันละ 2 ครั้ง ด้วยคลีนเซอร์อ่อนโยน หลีกเลี่ยงการขัดถูแรง ๆ
2) หลีกเลี่ยงการบีบ แกะ แกะเกา เพราะทำให้เป็นรอยแดง รอยดำ และแผลเป็นได้
3) ใช้ผลิตภัณฑ์ที่ไม่อุดตันรูขุมขน (non-comedogenic) และหลีกเลี่ยงเครื่องสำอาง/ครีมที่มันมาก
4) นอนพักผ่อนให้พอ ลดความเครียด ซึ่งเป็นปัจจัยกระตุ้นสิว
5) หากสิวอักเสบมาก เป็นก้อน เจ็บ หรือเป็นเรื้อรัง ควรพบแพทย์ผิวหนังเพื่อประเมินการใช้ยาทา/ยากินที่เหมาะสม

*หากมีสิวอักเสบเป็นหนองจำนวนมาก เจ็บมาก หรือเริ่มมีแผลเป็นชัด ควรพบแพทย์เพื่อตรวจและรักษาเพิ่มเติมนะคะ*`,
  },
  // ... (ส่วน diseaseInfo อื่น ๆ เหมือนของเดิมทั้งหมด ไม่ตัดทิ้งนะคะ)
  // เพื่อไม่ให้คำตอบยาวเกิน ใส่ของเดิมเดียร์กลับไปทั้งก้อนได้เลย
  Bullous: { /* ... */ },
  Chickenpox: { /* ... */ },
  Eczema: { /* ... */ },
  Psoriasis: { /* ... */ },
  Shingles: { /* ... */ },
  Warts: { /* ... */ },
  NormalSkin: { /* ... */ },
  Unknown: { /* ... */ },
};

// mapping entity → ชื่อไทยสั้น ๆ ใช้ตอนแสดงผล
const diseaseEntityToTh = {
  Acne: "สิว",
  Bullous: "ตุ่มน้ำพอง",
  Chickenpox: "อีสุกอีใส",
  Eczema: "ผิวหนังอักเสบ",
  Psoriasis: "สะเก็ดเงิน",
  Shingles: "งูสวัด",
  Warts: "หูด",
  NormalSkin: "ผิวปกติ",
  Unknown: "ไม่สามารถระบุได้",
};

// ====== Mapping: Thai disease name -> Entity key (ใช้กับ Rules sheet) ======
const diseaseThToEntity = {
  "สิว": "Acne",
  "สิวอุดตัน": "Acne",
  "สิวอักเสบ": "Acne",
  "ตุ่มน้ำพอง": "Bullous",
  "ตุ่มน้ำพองจากภูมิคุ้มกัน": "Bullous",
  "อีสุกอีใส": "Chickenpox",
  "สุกใส": "Chickenpox",
  "ผิวหนังอักเสบ": "Eczema",
  "ผิวหนังอักเสบเอ็กซีมา": "Eczema",
  "เอ็กซีมา": "Eczema",
  "สะเก็ดเงิน": "Psoriasis",
  "งูสวัด": "Shingles",
  "หูด": "Warts",
  "หูดฝ่าเท้า": "Warts",
  "ผิวปกติ": "NormalSkin",
  "ผิวไม่เป็นโรค": "NormalSkin",
  "ไม่สามารถระบุได้": "Unknown",
  "ไม่ทราบ": "Unknown",
};

// ====== โหลด labels + model ======
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

// ===== Helper: ตอบ LINE =====
async function replyMessage(replyToken, messages) {
  let msgs;

  if (Array.isArray(messages)) {
    // กรณีส่งมาเป็น array อยู่แล้ว
    msgs = messages;
  } else if (messages && typeof messages === "object" && messages.type) {
    // กรณีส่ง object เดียว เช่น { type: "image", ... }
    msgs = [messages];
  } else {
    // กรณีเป็น string ธรรมดา
    msgs = [{ type: "text", text: String(messages) }];
  }

  try {
    await axios.post(
      "https://api.line.me/v2/bot/message/reply",
      { replyToken, messages: msgs },
      {
        headers: { Authorization: `Bearer ${LINE_ACCESS_TOKEN}` },
        timeout: 15000,
      }
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
  let b = [-1, -1],
    s = [-1, -1];
  for (let i = 0; i < ps.length; i++) {
    const p = ps[i];
    if (p > b[1]) {
      s = b;
      b = [i, p];
    } else if (p > s[1]) {
      s = [i, p];
    }
  }
  return { bestIdx: b[0], bestProb: b[1], secondIdx: s[0], secondProb: s[1] };
}
function softmax(arr) {
  const m = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / s);
}
function sharpenProbs(probs, gamma) {
  const raised = probs.map((p) => Math.pow(Math.max(p, 1e-12), gamma));
  const s = raised.reduce((a, b) => a + b, 0);
  return raised.map((v) => v / s);
}

// ===== Core: classify =====
async function classifyImage(imageBuffer, { debug = false } = {}) {
  if (!modelReady) throw new Error("Model not ready");

  let pre;
  try {
    pre = await sharp(imageBuffer, { limitInputPixels: false })
      .rotate()
      .ensureAlpha()
      .removeAlpha()
      .resize(INPUT_SIZE, INPUT_SIZE, { fit: "cover" })
      .toFormat("png")
      .toBuffer();
  } catch (e) {
    throw new Error("IMAGE_PREPROCESS_FAIL: " + e.message);
  }

  let x = tf.node.decodeImage(pre, 3).toFloat().expandDims(0);
  if (!MODEL_INCLUDES_RESCALE) x = x.div(255);

  let y = null;
  try {
    y = model.predict ? model.predict(x) : null;
    if (Array.isArray(y)) y = y[0];
    if (!y || typeof y.dataSync !== "function") {
      const feedName = model.inputs?.[0]?.name;
      const fetchName = model.outputs?.[0]?.name;
      y = model.execute(feedName ? { [feedName]: x } : x, fetchName);
    }
  } catch (e) {
    tf.dispose(x);
    throw new Error("MODEL_EXEC_FAIL: " + e.message);
  }

  let probs;
  try {
    const raw = Array.from(y.dataSync());
    const sum = raw.reduce((p, c) => p + c, 0);
    probs = Math.abs(sum - 1) > 1e-3 ? softmax(raw) : raw;

    const nameByIdx = labels;
    const weighted = probs.map((p, i) => {
      const name = nameByIdx[i] || "";
      const w = CALIB_WEIGHTS[name] || 1.0;
      return p * w;
    });
    const s = weighted.reduce((a, b) => a + b, 0);
    probs = weighted.map((v) => v / (s || 1));

    probs = sharpenProbs(probs, PROB_SHARPEN_GAMMA);
  } catch (e) {
    tf.dispose([x, y]);
    throw new Error("POSTPROCESS_FAIL: " + e.message);
  }

  const { bestIdx, bestProb, secondProb } = top2(probs);
  const bestName = labels[bestIdx] || "Unknown";
  const ent = entropy(probs);

  const perClassTh = PER_CLASS_THRESHOLDS[bestName];

  const isUnknown =
    bestProb < (perClassTh ?? UNKNOWN_THRESHOLD) ||
    bestProb - secondProb < MARGIN_THRESHOLD ||
    ent > ENTROPY_THRESHOLD;

  const idx = isUnknown ? labels.length - 1 : bestIdx;
  const label = labels[idx] || "Unknown";
  const score = Number((bestProb * 100).toFixed(2));

  if (debug) {
    const top3 = [...probs]
      .map((p, i) => ({ i, p }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 3)
      .map((o) => `${labels[o.i]}:${(o.p * 100).toFixed(1)}%`)
      .join(", ");
    console.log(
      `[DEBUG] top3 = ${top3} | H=${ent.toFixed(3)} | margin=${(bestProb - secondProb).toFixed(
        3
      )}`
    );
  }

  tf.dispose([x, y]);
  return { label, score, appliedUnknown: isUnknown };
}

// ===== Dialogflow Setup =====
const DIALOGFLOW_PROJECT_ID = process.env.DIALOGFLOW_PROJECT_ID;
let dfSessionsClient = null;

if (DIALOGFLOW_PROJECT_ID) {
  const dfOptions = {};
  if (process.env.GOOGLE_CREDS_JSON) {
    dfOptions.credentials = JSON.parse(process.env.GOOGLE_CREDS_JSON);
  }
  dfSessionsClient = new dialogflow.SessionsClient(dfOptions);
}

async function detectIntent(sessionId, text) {
  if (!dfSessionsClient || !DIALOGFLOW_PROJECT_ID) {
    throw new Error("Dialogflow not configured");
  }
  const sessionPath = dfSessionsClient.projectAgentSessionPath(
    DIALOGFLOW_PROJECT_ID,
    sessionId
  );

  const request = {
    session: sessionPath,
    queryInput: {
      text: {
        text,
        languageCode: "th",
      },
    },
  };

  const [response] = await dfSessionsClient.detectIntent(request);
  return response.queryResult;
}

// ===== Google Sheets Setup =====
const QUESTIONS_SHEET_ID = process.env.QUESTIONS_SHEET_ID || process.env.SHEETS_ID;
const QUESTIONS_RANGE = process.env.QUESTIONS_RANGE || "derma_questions!A1:Z500";

const RULES_SHEET_ID = process.env.RULES_SHEET_ID || process.env.RULES_ID;
const RULES_RANGE = process.env.RULES_RANGE || "Rules!A1:D500";

let sheetsApi = null;
async function getSheetsApi() {
  if (sheetsApi) return sheetsApi;
  const auth = new google.auth.GoogleAuth({
    credentials: JSON.parse(process.env.GOOGLE_CREDS_JSON),
    scopes: ["https://www.googleapis.com/auth/spreadsheets.readonly"],
  });
  sheetsApi = google.sheets({ version: "v4", auth });
  return sheetsApi;
}

// ===== Quiz: โหลดคำถาม =====
let questionsCache = null;

async function loadQuestions() {
  if (questionsCache) return questionsCache;
  if (!QUESTIONS_SHEET_ID) throw new Error("QUESTIONS_SHEET_ID not set");

  const sheets = await getSheetsApi();
  const res = await sheets.spreadsheets.values.get({
    spreadsheetId: QUESTIONS_SHEET_ID,
    range: QUESTIONS_RANGE,
  });

  const rows = res.data.values || [];
  if (!rows.length) {
    questionsCache = [];
    return questionsCache;
  }

  const dataRows = rows.slice(1); // ข้าม header

  // header: qid | question_th | type | options (A|B|C|...)
  questionsCache = dataRows
    .filter((r) => r[0] && r[1])
    .map((r) => {
      const qid = r[0];
      const question = r[1];
      const type = r[2] || "choice";
      const optionsRaw = r[3] || "";
      const options = String(optionsRaw)
        .split("|")
        .map((s) => s.trim())
        .filter((s) => s);
      return { qid, question, type, options };
    });

  console.log("✅ Loaded questions:", questionsCache.length);
  return questionsCache;
}

// ===== Rules: โหลด qid | option_th | disease_th | score =====
let rulesCache = null;

async function loadRules() {
  if (rulesCache) return rulesCache;
  if (!RULES_SHEET_ID) throw new Error("RULES_SHEET_ID not set");

  const sheets = await getSheetsApi();
  const res = await sheets.spreadsheets.values.get({
    spreadsheetId: RULES_SHEET_ID,
    range: RULES_RANGE,
  });

  const rows = res.data.values || [];
  if (!rows.length) {
    rulesCache = [];
    return rulesCache;
  }

  const dataRows = rows.slice(1); // ข้าม header

  rulesCache = dataRows
    .filter((r) => r[0] && r[1] && r[2])
    .map((r) => ({
      qid: r[0],
      option_th: r[1],
      disease_th: r[2],
      score: Number(r[3] || 0),
    }));

  console.log("✅ Loaded rules (rows):", rulesCache.length);
  return rulesCache;
}

// ===== คำนวณโรคจากคำตอบ quiz ตาม rules =====
async function calculateDiseaseFromRules(answers) {
  const rules = await loadRules();

  const diseaseEntities = [
    "Acne",
    "Bullous",
    "Chickenpox",
    "Eczema",
    "Psoriasis",
    "Shingles",
    "Warts",
    "NormalSkin",
    "Unknown",
  ];

  const scores = {};
  diseaseEntities.forEach((k) => {
    scores[k] = 0;
  });

  for (const rule of rules) {
    const ans = answers[rule.qid];
    if (!ans) continue;

    if (String(ans.optionText).trim() === String(rule.option_th).trim()) {
      const entityKey = diseaseThToEntity[rule.disease_th] || "Unknown";
      const addScore = Number(rule.score || 0);
      scores[entityKey] += addScore;
    }
  }

  let bestDiseaseEntity = "Unknown";
  let bestScore = 0;
  for (const d of diseaseEntities) {
    if (scores[d] > bestScore) {
      bestScore = scores[d];
      bestDiseaseEntity = d;
    }
  }

  if (bestScore <= 0) bestDiseaseEntity = "Unknown";

  return { bestDiseaseEntity, scores, bestScore };
}

// ===== Quiz state per user =====
const quizState = new Map();

// quick reply ปุ่มคำตอบแต่ละข้อ
function buildQuestionMessages(qIndex, total, q) {
  const header = `ข้อที่ ${qIndex + 1}/${total}\n${q.question}`;

  // แสดงตัวเลือกแบบเต็มในข้อความหลัก
  const optionLines = q.options
    .map((opt, i) => `${i + 1}) ${opt}`)
    .join("\n");

  const quickItems = q.options.map((opt, i) => ({
    type: "action",
    action: {
      type: "message",
      // ใช้แค่เลขเป็น label ไม่เกิน 20 ตัวแน่นอน
      label: String(i + 1),
      text: String(i + 1), // ผู้ใช้กดแล้วส่งเลขกลับมา
    },
  }));

  return [
    {
      type: "text",
      text: `${header}\n\n${optionLines}`,
      quickReply: {
        items: quickItems,
      },
    },
  ];
}


async function startQuizForUser(userId, replyToken) {
  const questions = await loadQuestions();
  if (!questions.length) {
    await replyMessage(replyToken, "ขออภัย ระบบยังไม่มีคำถามให้ทำแบบประเมินค่ะ");
    return;
  }

  quizState.set(userId, {
    inProgress: true,
    currentIndex: 0,
    questions,
    answers: {},
  });

  const firstQ = questions[0];
  await replyMessage(replyToken, buildQuestionMessages(0, questions.length, firstQ));
}

async function handleQuizAnswer(userId, replyToken, userText) {
  const state = quizState.get(userId);
  if (!state || !state.inProgress) {
    await replyMessage(
      replyToken,
      'หากต้องการเริ่มทำแบบประเมินใหม่ พิมพ์ว่า "เริ่มแบบประเมินผิวหนัง" ได้เลยนะคะ หรือกดปุ่มในเมนูด้านล่างค่ะ 😊'
    );
    return;
  }

  const q = state.questions[state.currentIndex];
  const total = state.questions.length;

  const num = parseInt(userText.trim(), 10);
  if (Number.isNaN(num) || num < 1 || num > q.options.length) {
    const msgs = buildQuestionMessages(state.currentIndex, total, q);
    msgs[0].text = "กรุณาเลือกคำตอบจากปุ่มด้านล่างนะคะ 😊\n\n" + msgs[0].text;
    await replyMessage(replyToken, msgs);
    return;
  }

  const idx = num - 1;
  const optionText = q.options[idx];

  state.answers[q.qid] = { optionIndex: idx, optionText };

  if (state.currentIndex + 1 < total) {
    state.currentIndex += 1;
    const nextQ = state.questions[state.currentIndex];
    await replyMessage(replyToken, buildQuestionMessages(state.currentIndex, total, nextQ));
  } else {
    state.inProgress = false;

    const { bestDiseaseEntity, bestScore } = await calculateDiseaseFromRules(
      state.answers
    );

    const thName = diseaseEntityToTh[bestDiseaseEntity] || bestDiseaseEntity;
    const infoObj = diseaseInfo[bestDiseaseEntity] || diseaseInfo.Unknown;

    await replyMessage(replyToken, [
      {
        type: "text",
        text:
          `สรุปผลจากแบบประเมิน 15 ข้อ\n` +
          `อาการของคุณเข้าได้มากที่สุดกับ: ${thName}\n(คะแนนรวม: ${bestScore})`,
      },
      { type: "text", text: infoObj.info },
      { type: "text", text: infoObj.care },
      { type: "text", text: DISCLAIMER },
    ]);
  }
}

// ===== LINE Webhook =====
app.post("/webhook", async (req, res) => {
  const events = req.body?.events || [];
  for (const event of events) {
    const replyToken = event.replyToken;
    const userId = event.source?.userId || "unknown-user";

    try {
      if (event.type === "message") {
        // ==== IMAGE ====
        if (event.message.type === "image") {
          if (!modelReady) {
            await replyMessage(
              replyToken,
              "โมเดลกำลังโหลดอยู่ กรุณาลองอีกครั้งในไม่กี่วินาทีค่ะ"
            );
            continue;
          }

          let imgBuf;
          try {
            const r = await axios.get(
              `https://api-data.line.me/v2/bot/message/${event.message.id}/content`,
              {
                headers: {
                  Authorization: `Bearer ${LINE_ACCESS_TOKEN}`,
                },
                responseType: "arraybuffer",
                timeout: 30000,
                maxContentLength: Infinity,
                maxBodyLength: Infinity,
              }
            );
            imgBuf = Buffer.from(r.data);
          } catch (e) {
            console.error("Fetch image error:", e?.response?.status || e.message);
            await replyMessage(
              replyToken,
              "ดึงรูปจาก LINE ไม่สำเร็จ ลองส่งใหม่เป็น JPG/PNG ดูนะคะ"
            );
            continue;
          }

          try {
            const { label, score, appliedUnknown } = await classifyImage(imgBuf, {
              debug: false,
            });

            const diseaseKey = diseaseInfo[label]
              ? label
              : label === "NormalSkin"
              ? "NormalSkin"
              : "Unknown";

            const thName = diseaseEntityToTh[diseaseKey] || diseaseKey;
            const infoObj = diseaseInfo[diseaseKey] || diseaseInfo.Unknown;

            const extra = appliedUnknown ? " (จัดเป็น Unknown/ไม่มั่นใจ)" : "";

            await replyMessage(replyToken, [
              {
                type: "text",
                text:
                  `ผลการจำแนกจากรูปภาพ:\n` +
                  `คาดว่าเป็น: ${thName}${extra}\n` +
                  `ความเชื่อมั่นของโมเดล (class สูงสุด) ≈ ${score.toFixed(1)}%`,
              },
              { type: "text", text: infoObj.info },
              { type: "text", text: infoObj.care },
              { type: "text", text: DISCLAIMER },
            ]);
          } catch (e) {
            console.error("Classify error:", e.message);
            await replyMessage(
              replyToken,
              "ประมวลผลภาพไม่สำเร็จ กรุณาลองใหม่อีกครั้งค่ะ (ลองส่งเป็น JPG/PNG ขนาดไม่ใหญ่เกินไป)"
            );
          }

          continue;
        }

        // ==== TEXT ====
        if (event.message.type === "text") {
          const text = event.message.text || "";
          const normalizedText = text.replace(/\s+/g, " ").trim();

          console.log("TEXT FROM USER:", JSON.stringify(text));

          // 1) ปุ่ม Rich menu: คู่มือการใช้งาน
          if (
            normalizedText === USER_GUIDE_KEYWORD ||
            normalizedText === "คู่มือการใช้งาน"
          ) {
            await replyMessage(replyToken, {
              type: "image",
              originalContentUrl: USER_GUIDE_IMAGE_URL,
              previewImageUrl: USER_GUIDE_IMAGE_URL,
            });
            continue;
          }

          // 2) ปุ่ม Rich menu: โรคผิวหนังที่กำลังระบาด
          if (
            normalizedText === OUTBREAK_KEYWORD ||
            normalizedText === "โรคผิวหนังที่กำลังระบาด"
          ) {
            await replyMessage(replyToken, {
              type: "image",
              originalContentUrl: OUTBREAK_IMAGE_URL,
              previewImageUrl: OUTBREAK_IMAGE_URL,
            });
            continue;
          }

          // 3) เริ่ม quiz จากปุ่มเมนู A
          if (
            normalizedText === START_QUIZ_KEYWORD ||
            normalizedText === "เริ่มแบบประเมิน" ||
            normalizedText === "เริ่มทำแบบประเมินผิวหนัง" ||
            normalizedText === "เริ่มทำแบบประเมินผื่นผิวหนัง"
          ) {
            await startQuizForUser(userId, replyToken);
            continue;
          }

          // 4) ถ้ากำลังทำ quiz อยู่ → ตีความข้อความเป็นคำตอบ (เลขจากปุ่ม quick reply)
          const state = quizState.get(userId);
          if (state && state.inProgress) {
            await handleQuizAnswer(userId, replyToken, normalizedText);
            continue;
          }

          // 5) ถามข้อมูลทั่วไป → แสดงโรคที่ถามได้ (ไม่เข้า Dialogflow)
if (
  normalizedText === "พิมพ์เพื่อสอบถามข้อมูล/การดูแลโรคผิวหนังเบื้องต้น" ||
  normalizedText === "สอบถามข้อมูลโรคผิวหนัง" ||
  normalizedText === "ข้อมูลโรคผิวหนัง"
) {
  await replyMessage(replyToken, {
    type: "text",
    text:
      "คุณสามารถพิมพ์ชื่อโรคผิวหนังเพื่อดูข้อมูลและการดูแลเบื้องต้นได้นะคะ 😊\n\n" +
      "ตัวอย่างเช่น:\n" +
      "• สิว\n" +
      "• ผิวหนังอักเสบ (เอ็กซีมา)\n" +
      "• งูสวัด\n" +
      "• สะเก็ดเงิน\n\n" +
      "...หรือโรคอื่น ๆ ที่สนใจได้เลยค่ะ",
  });
  continue;
}

          // 6) ถามตอบทั่วไป → ส่งเข้า Dialogflow
          try {
            const result = await detectIntent(userId, text);
            const params = result.parameters?.fields || {};

            let diseaseParam = "";
            let askType = "both"; // ค่า default

            if (params.disease) {
              diseaseParam = params.disease.stringValue || "";
            }
            if (params.ask_type) {
              askType = params.ask_type.stringValue || "both";
            }

            if (diseaseParam && diseaseInfo[diseaseParam]) {
              const thName = diseaseEntityToTh[diseaseParam] || diseaseParam;
              const infoObj = diseaseInfo[diseaseParam];

              const msgs = [{ type: "text", text: `ข้อมูลเกี่ยวกับ: ${thName}` }];

              // ส่ง info ตาม askType
              if (askType === "info" || askType === "both") {
                msgs.push({ type: "text", text: infoObj.info });
              }

              // ส่ง care ตาม askType
              if (askType === "care" || askType === "both") {
                msgs.push({ type: "text", text: infoObj.care });
              }

              // ปิดท้ายด้วย DISCLAIMER ทุกครั้งที่มีการให้คำแนะนำเกี่ยวกับโรค
              msgs.push({ type: "text", text: DISCLAIMER });

              await replyMessage(replyToken, msgs);
            } else {
              const fallback =
                result.fulfillmentText ||
                "คุณสามารถส่งรูปผื่นผิวหนัง หรือถามเกี่ยวกับโรคสิว ผื่น ลมพิษ งูสวัด ฯลฯ ได้เลยนะคะ";
              await replyMessage(replyToken, fallback);
            }
          } catch (e) {
            console.error("Dialogflow error:", e.message);
            await replyMessage(
              replyToken,
              "ตอนนี้ไม่สามารถเชื่อมต่อระบบตอบคำถามได้ชั่วคราว แต่ยังส่งรูปให้ช่วยประเมินผื่นได้นะคะ"
            );
          }

          continue;
        }

        // type message อื่น ๆ
        await replyMessage(replyToken, "ตอนนี้รองรับเฉพาะข้อความและรูปภาพนะคะ");
      } else {
        await replyMessage(replyToken, "ยังรองรับเฉพาะข้อความและรูปภาพนะคะ");
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
app.get("/healthz", (_req, res) =>
  res.json({
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
      PER_CLASS_THRESHOLDS,
    },
  })
);

app.post(
  "/debug/classify",
  express.raw({ type: "*/*", limit: "10mb" }),
  async (req, res) => {
    try {
      const out = await classifyImage(req.body, {
        debug: true,
      });
      res.json(out);
    } catch (e) {
      res.status(500).json({ error: e.message });
    }
  }
);

app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
