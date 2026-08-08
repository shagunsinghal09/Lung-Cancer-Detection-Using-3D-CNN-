const form = document.getElementById('upload-form');
const result = document.getElementById('result');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  result.classList.add('hidden');

  const data = new FormData(form);
  const response = await fetch('/predict', { method: 'POST', body: data });
  const payload = await response.json();

  if (!response.ok) {
    result.innerHTML = `<h3>Error</h3><p>${payload.error}</p>`;
  } else {
    result.innerHTML = `
      <h3>Prediction Result</h3>
      <p><strong>Label:</strong> ${payload.label}</p>
      <p><strong>Probability:</strong> ${payload.probability}</p>
      <p><strong>Confidence:</strong> ${payload.confidence}</p>
      <p>${payload.disclaimer}</p>
    `;
  }
  result.classList.remove('hidden');
});
