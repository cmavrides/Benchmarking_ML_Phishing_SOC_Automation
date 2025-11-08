const form = document.getElementById("classify-form");
const resultSection = document.getElementById("result");
const labelSpan = document.getElementById("label");
const scoreSpan = document.getElementById("score");
const riskSpan = document.getElementById("risk");
const confidenceSpan = document.getElementById("confidence");
const rationaleDiv = document.getElementById("rationale");
const recommendationsDiv = document.getElementById("recommendations");
const iocsDiv = document.getElementById("iocs");
const explanationsDiv = document.getElementById("explanations");

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
    scoreSpan.textContent = typeof data.score === "number" ? data.score.toFixed(4) : "N/A";
    riskSpan.textContent = data.risk_level ? data.risk_level.toUpperCase() : "N/A";
    confidenceSpan.textContent =
      typeof data.confidence === "number" ? `${(data.confidence * 100).toFixed(1)}%` : "N/A";

    if (data.explanations && data.explanations.rationale) {
      rationaleDiv.innerHTML = `<p><strong>Rationale:</strong> ${data.explanations.rationale}</p>`;
    } else {
      rationaleDiv.innerHTML = "";
    }

    if (Array.isArray(data.recommendations) && data.recommendations.length > 0) {
      const items = data.recommendations.map((item) => `<li>${item}</li>`).join("");
      recommendationsDiv.innerHTML = `<h3>Suggested Actions</h3><ul>${items}</ul>`;
    } else {
      recommendationsDiv.innerHTML = "";
    }

    if (data.iocs && Object.keys(data.iocs).length > 0) {
      const parts = Object.entries(data.iocs)
        .map(([key, values]) => `<p><strong>${key.toUpperCase()}:</strong> ${values.join(", ")}</p>`)
        .join("");
      const summary = data.ioc_summary && data.ioc_summary.total ? `<p><em>Total IOCs detected: ${data.ioc_summary.total}</em></p>` : "";
      iocsDiv.innerHTML = `<h3>Indicators of Compromise</h3>${summary}${parts}`;
    } else {
      iocsDiv.innerHTML = "<p>No IOCs detected.</p>";
    }

    const explanation = data.explanations || {};
    const explanationSections = [];
    if (Array.isArray(explanation.supporting_terms) && explanation.supporting_terms.length > 0) {
      const items = explanation.supporting_terms
        .map((item) => `<li>${item.term} <small>(${item.contribution.toFixed(3)})</small></li>`)
        .join("");
      explanationSections.push(`<div><h3>Phishing cues</h3><ul>${items}</ul></div>`);
    }
    if (Array.isArray(explanation.mitigating_terms) && explanation.mitigating_terms.length > 0) {
      const items = explanation.mitigating_terms
        .map((item) => `<li>${item.term} <small>(${item.contribution.toFixed(3)})</small></li>`)
        .join("");
      explanationSections.push(`<div><h3>Legitimacy cues</h3><ul>${items}</ul></div>`);
    }
    if (!explanationSections.length && Array.isArray(explanation.top_terms) && explanation.top_terms.length > 0) {
      const items = explanation.top_terms
        .map((item) => `<li>${item.term} <small>(${item.weight.toFixed(3)})</small></li>`)
        .join("");
      explanationSections.push(`<div><h3>Salient terms</h3><ul>${items}</ul></div>`);
    }

    if (explanationSections.length > 0) {
      const method =
        typeof explanation.method === "string"
          ? `<p><em>Explanation method: ${explanation.method.replace(/_/g, " ")}</em></p>`
          : "";
      explanationsDiv.innerHTML = `<h3>Model Explainability</h3>${method}${explanationSections.join("")}`;
    } else {
      explanationsDiv.innerHTML = "";
    }

    resultSection.hidden = false;
  } catch (error) {
    alert(`Error: ${error.message}`);
  }
}

form.addEventListener("submit", classify);
