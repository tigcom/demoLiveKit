import { createLocalAudioTrack, Room } from "livekit-client";

// Load LiveKit URL from environment variables (Vite)
const LIVEKIT_URL = import.meta.env.VITE_LIVEKIT_URL || "wss://localhost:7880";
const TOKEN_SERVER_URL = import.meta.env.VITE_TOKEN_SERVER_URL || "http://localhost:8000";

const logsEl = document.getElementById("logs");
const logs = (m) => {
  logsEl.innerText += m + "\n";
  console.log(m);
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
  
  try {
    const j = await res.json();
    return j;
  } catch (e) {
    throw new Error(`Invalid token response (not JSON): ${e.message}`);
  }
}

document.getElementById("join").addEventListener("click", async () => {
  const roomName = document.getElementById("room").value.trim();
  const identity = document.getElementById("identity").value.trim();
  
  if (!roomName || !identity) {
    logs("❌ Error: Room and identity cannot be empty");
    return;
  }
  
  logs("➜ Requesting token...");
  let tokenResp;
  try {
    tokenResp = await getToken(roomName, identity);
  } catch (e) {
    logs(`❌ Token request failed: ${e.message}`);
    logs("💡 Check: Is token_server running on port 8000?");
    logs("💡 Command: uvicorn token_server:app --port 8000");
    return;
  }
  
  if (!tokenResp.token || !tokenResp.wsUrl) {
    logs("❌ Invalid token response (missing token or wsUrl)");
    return;
  }

  const room = new Room();
  
  // Listen for connection state changes
  room.on("connectionStateChanged", (state) => {
    logs(`📡 Connection state: ${state}`);
  });
  
  try {
    logs(`➜ Connecting to ${tokenResp.wsUrl}...`);
    await room.connect(tokenResp.wsUrl, tokenResp.token);
    logs(`✓ Connected to room: ${room.name}`);
    logs(`✓ Local participant: ${room.localParticipant.identity}`);
  } catch (e) {
    logs(`❌ Connection failed: ${e.message}`);
    logs(`💡 Check: VITE_LIVEKIT_URL in frontend .env is correct`);
    logs(`💡 Check: Token server is running (${TOKEN_SERVER_URL})`);
    logs(`💡 Details: ${e.toString()}`);
    return;
  }

  // Publish mic
  try {
    logs("➜ Publishing local microphone...");
    const track = await createLocalAudioTrack();
    await room.localParticipant.publishTrack(track);
    logs("✓ Published local microphone");
  } catch (e) {
    logs(`❌ Failed to publish mic: ${e.message}`);
    logs("💡 Check: Microphone permissions granted?");
    return;
  }

  // Listen for remote participants
  room.on("participantConnected", (p) => {
    logs(`👤 Participant connected: ${p.identity} (${p.kind})`);
  });
  
  room.on("participantDisconnected", (p) => {
    logs(`👤 Participant disconnected: ${p.identity}`);
  });

  // Subscribe to remote audio tracks
  room.on("trackSubscribed", (publication, track) => {
    logs(`📢 Track subscribed: ${publication.trackName || track.kind}`);
    
    if (track.kind === "audio") {
      const audioEl = document.createElement("audio");
      audioEl.autoplay = true;
      audioEl.controls = true;
      audioEl.style.display = "block";
      audioEl.style.marginTop = "1rem";
      
      // Attach media stream from track
      if (track.mediaStreamTrack) {
        const mediaStream = new MediaStream([track.mediaStreamTrack]);
        audioEl.srcObject = mediaStream;
      } else if (track.mediaStream) {
        audioEl.srcObject = track.mediaStream;
      } else {
        logs(`⚠️ Warning: Could not attach audio stream from track`);
        return;
      }
      
      document.body.appendChild(audioEl);
      logs("🔊 Audio track attached and playing automatically");
    }
  });
  
  room.on("trackUnsubscribed", (publication, track) => {
    logs(`📢 Track unsubscribed: ${publication.trackName || track.kind}`);
  });
  
  // Handle room disconnection
  room.on("disconnected", () => {
    logs("⚠️ Disconnected from room");
  });
});

