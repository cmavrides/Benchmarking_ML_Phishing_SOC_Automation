const form = document.getElementById("classify-form");
const resultSection = document.getElementById("result");
const labelSpan = document.getElementById("label");
const scoreSpan = document.getElementById("score");
const iocsDiv = document.getElementById("iocs");

async function classify(event) {
  event.preventDefault();
  const text = document.getElementById("text").value.trim();
  const isHtml = document.getElementById("is-html").checked;

  if (!text) {
    alert("Please provide text to classify.");
    return;
  }

  try {
    const response = await fetch("/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, is_html: isHtml }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Request failed");
    }

    const data = await response.json();
    labelSpan.textContent = data.label === 1 ? "Phishing" : "Legitimate";
    scoreSpan.textContent = data.score.toFixed(4);

    if (data.iocs && Object.keys(data.iocs).length > 0) {
      const parts = Object.entries(data.iocs)
        .map(([key, values]) => `<p><strong>${key.toUpperCase()}:</strong> ${values.join(", ")}</p>`)
        .join("");
      iocsDiv.innerHTML = `<h3>Indicators of Compromise</h3>${parts}`;
    } else {
      iocsDiv.innerHTML = "<p>No IOCs detected.</p>";
    }

    resultSection.hidden = false;
  } catch (error) {
    alert(`Error: ${error.message}`);
  }
}

form.addEventListener("submit", classify);
