require("dotenv").config();
const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
// const dialogflow = require("@google-cloud/dialogflow");
const FormData = require('form-data');

const app = express();
const PORT = process.env.PORT || 3000;
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;
const COLAB_API_URL = process.env.COLAB_API_URL;
const DIALOGFLOW_PROJECT_ID = process.env.DIALOGFLOW_PROJECT_ID;
// const CREDENTIALS = require("./dialogflow-key.json"); // ไฟล์ JSON ของ Dialogflow

app.use(bodyParser.json());

// เส้นทาง GET / เพื่อแสดงข้อความว่า webhook เชื่อมต่อสำเร็จ
app.get("/", (req, res) => {
    res.send("Webhook is working!");  // สามารถแก้ไขข้อความนี้ตามต้องการ
});

// ฟังก์ชันส่งข้อความกลับ LINE OA
async function replyMessage(replyToken, text) {
    await axios.post("https://api.line.me/v2/bot/message/reply", {
        replyToken: replyToken,
        messages: [{ type: "text", text: text }],
    }, {
        headers: { "Authorization": `Bearer ${LINE_ACCESS_TOKEN}` },
    });
}

// ฟังก์ชันเรียก Dialogflow เพื่อดึงข้อมูลโรค
async function getDiseaseInfo(diseaseName) {
    const sessionClient = new dialogflow.SessionsClient({ credentials: CREDENTIALS });
    const sessionPath = sessionClient.projectAgentSessionPath(DIALOGFLOW_PROJECT_ID, "12345");

    const request = {
        session: sessionPath,
        queryInput: {
            text: {
                text: diseaseName,
                languageCode: "th",
            },
        },
    };

    const responses = await sessionClient.detectIntent(request);
    return responses[0].queryResult.fulfillmentText;
}

// Webhook API รับภาพจาก LINE OA
app.post("/webhook", async (req, res) => {
    console.log("Request body: ", req.body);
    const events = req.body.events;
    for (let event of events) {
        if (event.type === "message" && event.message.type === "image") {
            const replyToken = event.replyToken;
            const imageId = event.message.id;

            try {
                // 1️⃣ ดึงรูปจาก LINE OA
                const imageBuffer = await axios({
                    method: "get",
                    url: `https://api-data.line.me/v2/bot/message/${imageId}/content`,
                    headers: { "Authorization": `Bearer ${LINE_ACCESS_TOKEN}` },
                    responseType: "arraybuffer",
                });

                // 2️⃣ ส่งรูปไปยัง Google Colab API
                const colabResponse = await axios.post(COLAB_API_URL, imageBuffer.data, {
                    headers: { "Content-Type": "application/octet-stream" },
                });
                console.log("Colab response:", colabResponse.data);  // เพิ่ม log ข้อมูลจาก Colab API
                
                const diseaseName = colabResponse.data.result || "ไม่สามารถจำแนกได้";

                // 3️⃣ ส่งชื่อโรคกลับไปที่ LINE OA
                await replyMessage(replyToken, `ผลการจำแนก: ${diseaseName}`);

                // 4️⃣ ส่งชื่อโรคไปที่ Dialogflow เพื่อดึงข้อมูลโรค
                // const diseaseInfo = await getDiseaseInfo(diseaseName);

                // 5️⃣ ส่งข้อมูลโรคกลับไปที่ LINE OA
                // await replyMessage(replyToken, `ข้อมูลโรค: ${diseaseInfo}`);
            } catch (error) {
                console.error("Error:", error);
                await replyMessage(replyToken, "เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง");
            }
        }
    }
    res.sendStatus(200);
});

// เริ่มเซิร์ฟเวอร์
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
