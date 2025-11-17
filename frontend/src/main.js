import { createLocalAudioTrack, Room } from "livekit-client";

const LIVEKIT_URL = import.meta.env.VITE_LIVEKIT_URL || "wss://localhost:7880";
const TOKEN_SERVER_URL = import.meta.env.VITE_TOKEN_SERVER_URL || "http://localhost:8000";

const logsEl = document.getElementById("logs");
const logs = (m) => {
  logsEl.innerText += m + "\n";
  console.log(m);
};

// Transcript container
let transcriptsEl = document.getElementById("transcripts");
if (!transcriptsEl) {
  transcriptsEl = document.createElement("div");
  transcriptsEl.id = "transcripts";
  transcriptsEl.style.border = "1px solid #ddd";
  transcriptsEl.style.padding = "0.5rem";
  transcriptsEl.style.marginTop = "1rem";
  transcriptsEl.style.maxHeight = "200px";
  transcriptsEl.style.overflowY = "auto";
  transcriptsEl.innerHTML = "<strong>Transcripts</strong><br/>";
  logsEl.parentNode.insertBefore(transcriptsEl, logsEl.nextSibling);
}

const appendTranscript = (who, text) => {
  const p = document.createElement("div");
  p.style.padding = "0.25rem 0";
  p.innerHTML = `<strong>${who}:</strong> ${text}`;
  transcriptsEl.appendChild(p);
  transcriptsEl.scrollTop = transcriptsEl.scrollHeight;
};

async function getToken(room, identity) {
  const res = await fetch(`${TOKEN_SERVER_URL}/api/get_token`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ room, identity }),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Token server error (${res.status}): ${errText}`);
  }

  return res.json();
}

document.getElementById("join").addEventListener("click", async () => {
  const roomName = document.getElementById("room").value.trim();
  const identity = document.getElementById("identity").value.trim();
  if (!roomName || !identity) return logs("❌ Room and identity cannot be empty");

  logs("➜ Requesting token...");
  let tokenResp;
  try {
    tokenResp = await getToken(roomName, identity);
  } catch (e) {
    return logs(`❌ Token request failed: ${e.message}`);
  }

  const room = new Room();
  room.on("connectionStateChanged", (state) => logs(`📡 Connection state: ${state}`));

  try {
    logs(`➜ Connecting to ${tokenResp.wsUrl}...`);
    await room.connect(tokenResp.wsUrl, tokenResp.token);
    logs(`✓ Connected: ${room.name}, Local participant: ${room.localParticipant.identity}`);
  } catch (e) {
    return logs(`❌ Connection failed: ${e.message}`);
  }

  // Publish mic
  let audioTrack;
  try {
    logs("➜ Publishing local microphone...");
    audioTrack = await createLocalAudioTrack();
    const pub = await room.localParticipant.publishTrack(audioTrack);
    console.log("📡 Published track:", pub);
    logs("✓ Local mic published");

    // Track mute/unmute events
    if (pub.track) {
      pub.track.on("mute", () => {
        logs("🔇 Track muted");
      });
      pub.track.on("unmute", () => {
        logs("🔊 Track unmuted");
      });
    }

    // --------------------------------------------------------
    // ✔ TEST 1: attach your own mic (self-monitor)
    // --------------------------------------------------------
    const selfAudioEl = document.createElement("audio");
    selfAudioEl.autoplay = true;
    selfAudioEl.controls = true;
    selfAudioEl.style.marginTop = "1rem";
    selfAudioEl.srcObject = new MediaStream([audioTrack.mediaStreamTrack]);
    document.body.appendChild(selfAudioEl);
    logs("🎧 Self-monitor enabled — you should hear your own mic");

    // --------------------------------------------------------
    // ✔ TEST 2: outbound audio stats
    // --------------------------------------------------------
    setInterval(async () => {
      const stats = await audioTrack.getRTCStatsReport();
      stats.forEach((report) => {
        if (report.type === "outbound-rtp" && report.kind === "audio") {
          console.log("🔊 packetsSent =", report.packetsSent);
          console.log("📦 bytesSent =", report.bytesSent);
          logs(`📢 Mic stats → packets=${report.packetsSent}, bytes=${report.bytesSent}`);
        }
      });
    }, 2000);

    // SpeechRecognition (optional)
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.lang = "vi-VN";
      recognition.continuous = true;
      recognition.interimResults = true;

      recognition.onresult = (event) => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        appendTranscript("Local User", transcript);
      };

      recognition.onerror = (event) => logs(`⚠️ SpeechRecognition error: ${event.error}`);
      recognition.start();
      logs("🎤 Listening to your voice...");
    } else {
      logs("⚠️ Browser does not support SpeechRecognition");
    }
  } catch (e) {
    return logs(`❌ Failed to publish mic: ${e.message}`);
  }

  // Remote participants
  room.on("participantConnected", (p) => logs(`👤 Participant connected: ${p.identity}`));
  room.on("participantDisconnected", (p) => logs(`👤 Participant disconnected: ${p.identity}`));

  // --------------------------------------------------------
  // ✔ TEST 3: verify trackPublished event
  // --------------------------------------------------------
  room.localParticipant.on("trackPublished", (pub) => {
    logs(`📡 trackPublished → ${pub.kind}`);
    console.log("Track published:", pub);
  });

  // Subscribe remote audio
  room.on("trackSubscribed", (publication, track) => {
    logs(`📢 Track subscribed: ${publication.trackName || track.kind}`);

    if (track.kind === "audio") {
      const audioEl = document.createElement("audio");
      audioEl.autoplay = true;
      audioEl.controls = true;
      audioEl.style.display = "block";
      audioEl.style.marginTop = "1rem";

      const mediaStream = new MediaStream([track.mediaStreamTrack]);
      audioEl.srcObject = mediaStream;
      document.body.appendChild(audioEl);
      logs("🔊 Playing remote audio stream");
    }
  });

  // Data messages
  room.on("dataReceived", (payload, participant) => {
    try {
      const raw = new TextDecoder().decode(
        payload instanceof Uint8Array ? payload : payload.data || payload
      );
      let msg;
      try {
        msg = JSON.parse(raw);
      } catch {
        msg = null;
      }

      if (!msg) return appendTranscript(participant?.identity || "remote", raw);

      if (msg.type === "transcript") {
        appendTranscript(msg.from || participant?.identity || "user", msg.text || "");
      } else if (msg.type === "greeting" && msg.url) {
        appendTranscript(msg.to || "agent", msg.text || "");
        const audioEl = document.createElement("audio");
        audioEl.autoplay = true;
        audioEl.controls = true;
        audioEl.src = msg.url;
        document.body.appendChild(audioEl);
        logs(`🔊 Playing greeting audio from ${msg.url}`);
      }
    } catch (e) {
      logs(`⚠️ Error handling dataReceived: ${e}`);
    }
  });

  room.on("disconnected", () => logs("⚠️ Disconnected from room"));
});
