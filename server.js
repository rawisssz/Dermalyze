require("dotenv").config();
const express = require("express");
const axios = require("axios");
const bodyParser = require("body-parser");
const dialogflow = require("@google-cloud/dialogflow"); // ต้องติดตั้ง package นี้
const FormData = require('form-data');

const app = express();
const PORT = process.env.PORT || 3000;
const LINE_ACCESS_TOKEN = process.env.LINE_ACCESS_TOKEN;
const COLAB_API_URL = process.env.COLAB_API_URL;
const DIALOGFLOW_PROJECT_ID = process.env.DIALOGFLOW_PROJECT_ID;
const CREDENTIALS = require("./dialogflow-key.json"); // ไฟล์ JSON ของ Dialogflow

app.use(bodyParser.json());

async function replyMessage(replyToken, text) {
    await axios.post("https://api.line.me/v2/bot/message/reply", {
        replyToken: replyToken,
        messages: [{ type: "text", text: text }],
    }, {
        headers: { "Authorization": `Bearer ${LINE_ACCESS_TOKEN}` },
    });
}

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
                }).catch(err => {
                    console.error('Error fetching image:', err);
                    return null;
                });

                if (!imageBuffer) {
                    await replyMessage(replyToken, "ไม่สามารถดึงรูปจาก LINE OA ได้");
                    return;
                }

                console.log("Image fetched successfully");

                // 2️⃣ ส่งรูปไปยัง Google Colab API
                const colabResponse = await axios.post(COLAB_API_URL, imageBuffer.data, {
                    headers: { "Content-Type": "application/octet-stream" },
                }).catch(err => {
                    console.error('Error sending image to Colab:', err);
                    return null;
                });

                if (!colabResponse || !colabResponse.data) {
                    console.error("No response from Colab API");
                    await replyMessage(replyToken, "ไม่สามารถส่งข้อมูลไปยัง Google Colab ได้");
                    return;
                }

                console.log("Colab response:", colabResponse.data);

                const diseaseName = colabResponse.data.result || "ไม่สามารถจำแนกได้";

                // 3️⃣ ส่งชื่อโรคกลับไปที่ LINE OA
                await replyMessage(replyToken, `ผลการจำแนก: ${diseaseName}`);

            } catch (error) {
                console.error("Error:", error);
                await replyMessage(replyToken, "เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง");
            }
        }
    }
    res.sendStatus(200);
});

app.get("/", (req, res) => {
    res.send("Webhook is working!");
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
