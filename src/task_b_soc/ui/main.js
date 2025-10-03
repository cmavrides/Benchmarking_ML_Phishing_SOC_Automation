const classifyBtn = document.getElementById("classifyBtn");
const resultEl = document.getElementById("result");
const statusEl = document.getElementById("status");

async function classify() {
  const subject = document.getElementById("subject").value;
  const body = document.getElementById("body").value;
  const asHtml = document.getElementById("asHtml").checked;

  if (!body.trim()) {
    statusEl.textContent = "Body is required.";
    return;
  }

  classifyBtn.disabled = true;
  statusEl.textContent = "Classifying…";

  try {
    const response = await fetch("/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject, body, as_html: asHtml }),
    });
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    const data = await response.json();
    resultEl.textContent = JSON.stringify(data, null, 2);
    statusEl.textContent = `Prediction: ${data.label === 1 ? "Phishing" : "Legitimate"} (score ${data.score.toFixed(3)})`;
  } catch (error) {
    statusEl.textContent = `Error: ${error.message}`;
  } finally {
    classifyBtn.disabled = false;
  }
}

classifyBtn.addEventListener("click", classify);
